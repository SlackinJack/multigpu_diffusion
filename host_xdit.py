import base64
import copy
import gc
import os
import requests
import torch
import torch._dynamo
import torch.distributed as dist
from diffusers import (
    AutoencoderKL,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    HunyuanDiT2DModel,
    QuantoConfig,
    SD3Transformer2DModel,
    Transformer2DModel,
    UNet2DConditionModel,
)
from diffusers.utils import load_image
from flask import Flask, request, jsonify
from PIL import Image
from transformers import (
    BertModel,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    LlamaModel,
    T5EncoderModel,
)


from xDiT.xfuser import (
    xFuserArgs,
    xFuserFluxPipeline,
    xFuserHunyuanDiTPipeline,
    xFuserPixArtAlphaPipeline,
    xFuserPixArtSigmaPipeline,
    xFuserStableDiffusion3Pipeline,
)
from xDiT.xfuser.config import FlexibleArgumentParser


from modules.host_generics import *
from modules.scheduler_config import *
from modules.utils import *


app = Flask(__name__)
engine_config = None
initialized = False
step_progress = 0
input_config = None
local_rank = None
logger = None
pipe = None
result = None
cache_args = {
    "use_teacache": True,
    "use_fbcache": True,
    "rel_l1_thresh": 0.12,
    "return_hidden_states_first": False,
    "num_steps": 30,
}


def get_args():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    # xDiT arguments
    """
        [--model MODEL] [--download-dir DOWNLOAD_DIR]
        [--trust-remote-code] [--warmup_steps WARMUP_STEPS]
        [--use_parallel_vae] [--use_torch_compile] [--use_onediff]
        [--use_teacache] [--use_fbcache] [--use_ray]
        [--ray_world_size RAY_WORLD_SIZE]
        [--dit_parallel_size DIT_PARALLEL_SIZE]
        [--use_cfg_parallel]
        [--data_parallel_degree DATA_PARALLEL_DEGREE]
        [--ulysses_degree ULYSSES_DEGREE]
        [--ring_degree RING_DEGREE]
        [--pipefusion_parallel_degree PIPEFUSION_PARALLEL_DEGREE]
        [--num_pipeline_patch NUM_PIPELINE_PATCH]
        [--attn_layer_num_for_pp [ATTN_LAYER_NUM_FOR_PP ...]]
        [--tensor_parallel_degree TENSOR_PARALLEL_DEGREE]
        [--vae_parallel_size VAE_PARALLEL_SIZE]
        [--split_scheme SPLIT_SCHEME] [--height HEIGHT]
        [--width WIDTH] [--num_frames NUM_FRAMES]
        [--img_file_path IMG_FILE_PATH] [--prompt [PROMPT ...]]
        [--no_use_resolution_binning]
        [--negative_prompt [NEGATIVE_PROMPT ...]]
        [--num_inference_steps NUM_INFERENCE_STEPS]
        [--max_sequence_length MAX_SEQUENCE_LENGTH] [--seed SEED]
        [--output_type OUTPUT_TYPE]
        [--guidance_scale GUIDANCE_SCALE]
        [--enable_sequential_cpu_offload]
        [--enable_model_cpu_offload] [--enable_tiling]
        [--enable_slicing] [--use_fp8_t5_encoder]
        [--use_fast_attn] [--n_calib N_CALIB]
        [--threshold THRESHOLD] [--window_size WINDOW_SIZE]
        [--coco_path COCO_PATH] [--use_cache]
    """
    #generic
    for k, v in GENERIC_HOST_ARGS.items():
        if k not in ["height", "width", "model"]:
            parser.add_argument(f"--{k}", type=v, default=None)
    for e in GENERIC_HOST_ARGS_TOGGLES:
        if e not in ["enable_model_cpu_offload", "enable_sequential_cpu_offload"]:
            parser.add_argument(f"--{e}", action="store_true")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    return args


@app.route("/initialize", methods=["GET"])
def check_initialize():
    global initialized
    if initialized: return "OK", 200
    else:           return "WAIT", 202


@app.route("/progress", methods=["GET"])
def check_progress():
    global step_progress
    return str(step_progress), 200


def initialize():
    global pipe, engine_config, input_config, local_rank, initialized, cache_args, logger
    args = get_args()

    # checks
    # TODO: checks

    # set torch type
    torch_dtype = get_torch_type(args.variant)

    # init distributed inference
    # remove all our args before passing it to xdit
    xargs = copy.deepcopy(args)
    del xargs.checkpoint
    del xargs.gguf_model
    del xargs.scheduler
    del xargs.warm_up_steps
    del xargs.port
    del xargs.variant
    del xargs.type
    del xargs.lora
    del xargs.compile_unet
    del xargs.compile_vae
    del xargs.compile_text_encoder
    del xargs.quantize_to
    del xargs.vae
    del xargs.control_net
    del xargs.ip_adapter
    del xargs.motion_adapter_lora
    del xargs.motion_adapter
    del xargs.motion_module
    engine_args = xFuserArgs.from_cli_args(xargs)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch_dtype
    local_rank = int(os.environ.get("LOCAL_RANK"))
    logger = get_logger(local_rank)
    # dynamo tweaks
    setup_torch_dynamo()
    # torch tweaks
    setup_torch_backends()

    # quantize
    quant = {}
    if args.quantize_to is not None:    quant = {"quantization_config": QuantoConfig(weights_dtype=args.quantize_to)}
    def quantize(model, desc):          do_quantization(model, desc, args.quantize_to, logger)

    # set pipeline
    logger.info(f"Initializing pipeline")
    kwargs = {}
    kwargs["engine_config"] = engine_config
    kwargs["cache_args"] = cache_args
    kwargs["torch_dtype"] = torch_dtype
    kwargs["use_safetensors"] = True
    kwargs["local_files_only"] = True
    kwargs["low_cpu_mem_usage"] = True

    kwargs_model = {}
    kwargs_model["torch_dtype"] = torch_dtype
    kwargs_model["use_safetensors"] = True
    kwargs_model["local_files_only"] = True
    kwargs_model["low_cpu_mem_usage"] = True

    PipelineClass = None
    to_quantize = {}
    quantize_unet_after = False

    match args.type:
        case "flux":
            if args.gguf_model is not None:
                kwargs["transformer"] = FluxTransformer2DModel.from_single_file(
                    args.gguf_model,
                    config=args.checkpoint,
                    subfolder="transformer",
                    torch_dtype=torch_dtype,
                    # use_safetensors=True,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                )
            if args.quantize_to is not None:
                if args.gguf_model is None:
                    if args.ip_adapter is None:
                        to_quantize["transformer"] = FluxTransformer2DModel.from_pretrained(args.checkpoint, subfolder="transformer", **quant, **kwargs_model)
                    else:
                        quantize_unet_after = True
                to_quantize["text_encoder"] = CLIPTextModel.from_pretrained(args.checkpoint, subfolder="text_encoder", **kwargs_model)
                to_quantize["text_encoder_2"] = T5EncoderModel.from_pretrained(args.checkpoint, subfolder="text_encoder_2", **kwargs_model)
            kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_model)
            PipelineClass = xFuserFluxPipeline
        case "hy":
            if args.quantize_to is not None:
                to_quantize["text_encoder"] = BertModel.from_pretrained(args.checkpoint, subfolder="text_encoder", **kwargs_model)
                to_quantize["text_encoder_2"] = T5EncoderModel.from_pretrained(args.checkpoint, subfolder="text_encoder_2", **kwargs_model)
                if args.ip_adapter is None:
                    to_quantize["transformer"] = HunyuanDiT2DModel.from_pretrained(args.checkpoint, subfolder="transformer", **quant, **kwargs_model)
                else:
                    quantize_unet_after = True
            kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_model)
            PipelineClass = xFuserHunyuanDiTPipeline
        case "pixa":
            if args.quantize_to is not None:
                to_quantize["text_encoder"] = T5EncoderModel.from_pretrained(args.checkpoint, subfolder="text_encoder", **kwargs_model)
                if args.ip_adapter is None:
                    to_quantize["transformer"] = Transformer2DModel.from_pretrained(args.checkpoint, subfolder="transformer", **quant, **kwargs_model)
                else:
                    quantize_unet_after = True
            kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_model)
            PipelineClass = xFuserPixArtAlphaPipeline
        case "pixs":
            if args.quantize_to is not None:
                to_quantize["text_encoder"] = T5EncoderModel.from_pretrained(args.checkpoint, subfolder="text_encoder", **kwargs_model)
                if args.ip_adapter is None:
                    to_quantize["transformer"] = Transformer2DModel.from_pretrained(args.checkpoint, subfolder="transformer", **quant, **kwargs_model)
                else:
                    quantize_unet_after = True
            kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_model)
            PipelineClass = xFuserPixArtSigmaPipeline
        case "sd3":
            if args.quantize_to is not None:
                to_quantize["text_encoder"] = CLIPTextModelWithProjection.from_pretrained(args.checkpoint, subfolder="text_encoder", **kwargs_model)
                to_quantize["text_encoder_2"] = CLIPTextModelWithProjection.from_pretrained(args.checkpoint, subfolder="text_encoder_2", **kwargs_model)
                to_quantize["text_encoder_3"] = T5EncoderModel.from_pretrained(args.checkpoint, subfolder="text_encoder_3", **kwargs_model)
                if args.ip_adapter is None:
                    to_quantize["transformer"] = SD3Transformer2DModel.from_pretrained(args.checkpoint, subfolder="transformer", **quant, **kwargs_model)
                else:
                    quantize_unet_after = True
            kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_model)
            PipelineClass = xFuserStableDiffusion3Pipeline
        case _: raise NotImplementedError

    # set vae
    if args.vae is not None:
        vae = AutoencoderKL.from_pretrained(args.vae, **quant, **kwargs_model)
        kwargs["vae"] = vae

    if len(to_quantize) > 0:
        for k, v in to_quantize.items():
            quantize(v, k)
            kwargs[k] = v

    pipe = PipelineClass.from_pretrained(args.checkpoint, **quant, **kwargs)
    logger.info(f"Pipeline initialized")

    # set ipadapter
    if args.ip_adapter is not None:
        args.ip_adapter = json.loads(args.ip_adapter)
        load_ip_adapter(pipe, args.ip_adapter)

    # deferred quantize
    if quantize_unet_after:
        quantize(pipe.transformer, "transformer")

    # set lora
    adapter_names = None
    if args.lora is not None:
        adapter_names = load_lora(args.lora, pipe, local_rank, logger, (args.quantize_to is not None))
        if len(to_quantize) > 0:
            logger.info("Requantizing unet/transformer, text encoder(s)")
            if to_quantize.get("unet") is not None:             quantize(pipe.unet, "unet")
            if to_quantize.get("transformer") is not None:      quantize(pipe.transformer, "transformer")
            if to_quantize.get("text_encoder") is not None:     quantize(pipe.text_encoder, "text_encoder")
            if to_quantize.get("text_encoder_2") is not None:   quantize(pipe.text_encoder_2, "text_encoder_2")
            if to_quantize.get("text_encoder_3") is not None:   quantize(pipe.text_encoder_3, "text_encoder_3")

    # set scheduler
    if args.scheduler is not None: pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)

    # set memory saving
    if args.type not in ["sd3"]:
        if args.enable_vae_slicing:         pipe.vae.enable_slicing()
        if args.enable_vae_tiling:          pipe.vae.enable_tiling()
    if args.xformers_efficient:             pipe.enable_xformers_memory_efficient_attention()
    if args.enable_sequential_cpu_offload:  pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
    elif args.enable_model_cpu_offload:     pipe.enable_model_cpu_offload(gpu_id=local_rank)
    else:                                   pipe = pipe.to(f"cuda:{local_rank}")

    # compiles
    if args.compile_unet:
        if args.type in "sdxl":         compile_unet(pipe, adapter_names, logger)
        elif args.gguf_model is None:   compile_transformer(pipe, adapter_names, logger)
    if args.compile_vae:                compile_vae(pipe, logger)
    if args.compile_text_encoder:       compile_text_encoder(pipe, logger)

    # set progress bar visibility
    pipe.set_progress_bar_config(disable=local_rank != 0)

    # clean up
    clean()

    # warm up run
    if args.warm_up_steps > 0:
        run_kwargs = {}
        run_kwargs["prompt"] = "a dog"
        run_kwargs["width"] = args.width
        run_kwargs["height"] = args.height
        run_kwargs["num_inference_steps"] = args.warm_up_steps
        run_kwargs["guidance_scale"] = 7.0
        run_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(1)
        run_kwargs["use_resolution_binning"] = input_config.use_resolution_binning
        run_kwargs["output_type"] = "pil"
        run_kwargs["max_sequence_length"] = 256
        if args.ip_adapter is not None:
            run_kwargs["ip_adapter_image"] = get_warmup_image()

        logger.info("Starting warmup run")
        pipe(**run_kwargs)

    # clean up
    clean()

    # complete
    logger.info("Model initialization completed")
    initialized = True
    return


def generate_image_parallel(
    dummy,
    positive,
    negative,
    positive_embeds,
    negative_embeds,
    ip_image,
    ip_image_scale,
    latent,
    steps,
    seed,
    cfg,
    clip_skip,
):
    global pipe, local_rank, input_config, result, step_progress
    args = get_args()
    torch.cuda.reset_peak_memory_stats()
    step_progress = 0

    if ip_image is not None and args.ip_adapter is not None:
        ip_image = load_image(ip_image)
        if ip_image_scale is not None and ip_image_scale != 100:
            percentage = ip_image_scale / 100
            ip_image = ip_image.resize((int(ip_image.size[0] * percentage), int(ip_image.size[1] * percentage)), Image.Resampling.LANCZOS)

    positive_pooled_embeds = None
    negative_pooled_embeds = None
    if positive_embeds is not None:
        positive_pooled_embeds = positive_embeds[0][1]["pooled_output"]
        positive_embeds = positive_embeds[0][0]
    if negative_embeds is not None:
        negative_pooled_embeds = negative_embeds[0][1]["pooled_output"]
        negative_embeds = negative_embeds[0][0]

    if latent is not None:
        latent = latent.to(get_torch_type(args.variant))

    def set_step_progress(pipe, index, timestep, callback_kwargs):
        global step_progress
        step_progress = index / steps * 100
        return callback_kwargs

    generator = torch.Generator(device="cpu").manual_seed(seed)

    is_image                                                    = True
    kwargs                                                      = {}
    kwargs["generator"]                                         = generator
    kwargs["guidance_scale"]                                    = cfg
    kwargs["num_inference_steps"]                               = steps
    kwargs["callback_on_step_end"]                              = set_step_progress
    kwargs["width"]                                             = args.width
    kwargs["height"]                                            = args.height
    kwargs["max_sequence_length"]                               = 256
    kwargs["output_type"]                                       = "pil"
    kwargs["use_resolution_binning"]                            = input_config.use_resolution_binning
    if positive is not None:                                    kwargs["prompt"]                    = positive
    if negative is not None:                                    kwargs["negative_prompt"]           = negative
    if positive_embeds is not None:                             kwargs["prompt_embeds"]             = positive_embeds
    if positive_pooled_embeds is not None:                      kwargs["pooled_prompt_embeds"]      = positive_pooled_embeds
    if negative_embeds is not None:                             kwargs["negative_embeds"]           = negative_embeds
    if negative_pooled_embeds is not None:                      kwargs["negative_pooled_embeds"]    = negative_pooled_embeds
    if latent is not None:                                      kwargs["latents"]                   = latent
    if clip_skip is not None:                                   kwargs["clip_skip"]                 = clip_skip
    if args.ip_adapter is not None:
        if ip_image is not None:
            kwargs["ip_adapter_image"] = ip_image
        else:
            return "No IPAdapter image provided for a IPAdapter-loaded pipeline", None, False

    if args.type in ["flux", "sd3"]:
        # TODO: fix callback/progressbar
        del kwargs["callback_on_step_end"]

    output = pipe(**kwargs)

    clean()

    if local_rank == 0:
        while True:
            if result is not None:
                step_progress = 100
                output_base64 = result
                result = None
                return "OK", output_base64, is_image
    elif output is not None:
        logger.info(f"task completed")
        if is_image:    o = output.images[0]
        else:           o = output.frames[0]
        output_base64 = pickle_and_encode_b64(o)
        with app.app_context():
            requests.post(f"http://localhost:{args.port}/set_result", json={"output": output_base64})


@app.route("/set_result", methods=["POST"])
def set_result():
    global result
    data = request.json
    result = data.get("output")
    return "", 200


@app.route("/generate", methods=["POST"])
def generate_image():
    global logger
    data = request.json
    dummy               = 0
    positive            = data.get("positive")
    negative            = data.get("negative")
    positive_embeds     = data.get("positive_embeds")
    negative_embeds     = data.get("negative_embeds")
    ip_image            = data.get("ip_image")
    ip_image_scale      = data.get("ip_image_scale")
    latent              = data.get("latent")
    steps               = data.get("steps")
    seed                = data.get("seed")
    cfg                 = data.get("cfg")
    clip_skip           = data.get("clip_skip")

    if positive is not None and len(positive) == 0:                     positive = None
    if negative is not None and len(negative) == 0:                     negative = None
    if image is None and positive is None and positive_embeds is None:  jsonify({ "message": "No input provided", "output": None, "is_image": False })
    if positive is not None and positive_embeds is not None:            jsonify({ "message": "Provide only one positive input", "output": None, "is_image": False })
    if negative is not None and negative_embeds is not None:            jsonify({ "message": "Provide only one negative input", "output": None, "is_image": False })
    if ip_image is not None:                                            ip_image = decode_b64_and_unpickle(ip_image)
    if latent is not None:                                              latent = decode_b64_and_unpickle(latent)
    if positive_embeds is not None:                                     positive_embeds = decode_b64_and_unpickle(positive_embeds)
    if negative_embeds is not None:                                     negative_embeds = decode_b64_and_unpickle(negative_embeds)

    params = [
        dummy,
        positive,
        negative,
        positive_embeds,
        negative_embeds,
        ip_image,
        ip_image_scale,
        latent,
        steps,
        seed,
        cfg,
        clip_skip
    ]
    dist.broadcast_object_list(params, src=0)
    print_params(
        {
            "positive": positive,
            "negative": negative,
            "positive_embeds": (positive_embeds is not None),
            "negative_embeds": (negative_embeds is not None),
            "ip_image": (ip_image is not None),
            "ip_image_scale": ip_image_scale,
            "latent": (latent is not None),
            "steps": steps,
            "seed": seed,
            "cfg": cfg,
            "clip_skip": clip_skip,
        }, logger)
    message, output_base64, is_image = generate_image_parallel(*params)
    response = { "message": message, "output": output_base64, "is_image": is_image }
    return jsonify(response)


def run_host():
    global logger
    args = get_args()
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        app.run(host="localhost", port=args.port)
    else:
        while True:
            params = [None] * 12 # len(params) of generate_image_parallel()
            logger.info(f"waiting for tasks")
            dist.broadcast_object_list(params, src=0)
            if params[0] is None:
                logger.info("Received exit signal, shutting down")
                break
            logger.info(f"Received task")
            generate_image_parallel(*params)


if __name__ == "__main__":
    initialize()
    run_host()
