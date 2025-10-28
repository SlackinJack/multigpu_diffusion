import base64
import copy
import gc
import os
import requests
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
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
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            global pipe, engine_config, input_config, local_rank, initialized, cache_args, logger
            args = get_args()

            # checks
            # TODO: checks

            # set torch type
            torch_dtype = get_torch_type(args.variant)

            # init distributed inference
            # remove all our args before passing it to xdit
            xargs = copy.copy(args)
            supported = [
                "height",
                "width",
                "xformers_efficient",
                "enable_model_cpu_offload",
                "enable_sequential_cpu_offload",
            ]

            remap = {
                "enable_vae_tiling": "enable_tiling",
                "enable_vae_slicing": "enable_slicing",
            }

            for k,v in GENERIC_HOST_ARGS.items():
                if k in xargs:
                    if k in remap.keys():
                        setattr(xargs, remap[k], v)
                        delattr(xargs, k)
                    elif k not in supported:
                        delattr(xargs, k)
            engine_args = xFuserArgs.from_cli_args(xargs)
            engine_config, input_config = engine_args.create_config()
            engine_config.runtime_config.dtype = torch_dtype
            local_rank = int(os.environ.get("LOCAL_RANK"))
            logger = get_logger(local_rank)
            # dynamo tweaks
            setup_torch_dynamo(args.torch_cache_limit, args.torch_accumlated_cache_limit, args.torch_capture_scalar)
            # torch tweaks
            setup_torch_backends()

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

            kwargs_vae = {}
            kwargs_vae["torch_dtype"] = torch.float32
            kwargs_vae["use_safetensors"] = True
            kwargs_vae["local_files_only"] = True
            kwargs_vae["low_cpu_mem_usage"] = True

            kwargs_gguf = {}
            kwargs_gguf["torch_dtype"] = torch_dtype
            kwargs_gguf["use_safetensors"] = False
            kwargs_gguf["local_files_only"] = True
            kwargs_gguf["low_cpu_mem_usage"] = True
            kwargs_gguf["quantization_config"] = GGUFQuantizationConfig(compute_dtype=torch_dtype)

            PipelineClass = None

            match args.type:
                case "flux":
                    if args.gguf_model is not None:
                        kwargs["transformer"] = FluxTransformer2DModel.from_single_file(args.gguf_model, config=args.checkpoint, subfolder="transformer", **kwargs_gguf)
                    kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_vae)
                    PipelineClass = xFuserFluxPipeline
                case "hy":
                    kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_vae)
                    PipelineClass = xFuserHunyuanDiTPipeline
                case "pixa":
                    kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_vae)
                    PipelineClass = xFuserPixArtAlphaPipeline
                case "pixs":
                    kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_vae)
                    PipelineClass = xFuserPixArtSigmaPipeline
                case "sd3":
                    kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_vae)
                    PipelineClass = xFuserStableDiffusion3Pipeline
                case _: raise NotImplementedError

            # set vae
            if args.vae is not None:
                kwargs["vae"] = AutoencoderKL.from_pretrained(args.vae, **kwargs_vae)

            # init pipe
            pipe = PipelineClass.from_pretrained(args.checkpoint, **kwargs)
            logger.info(f"Pipeline initialized")

            # set scheduler
            set_scheduler(args, pipe)

            # set ipadapter
            if args.ip_adapter is not None:
                args.ip_adapter = json.loads(args.ip_adapter)
                load_ip_adapter(pipe, args.ip_adapter)

            # set memory saving
            if args.type not in ["sd3"]:
                if args.enable_vae_slicing:         pipe.vae.enable_slicing()
                if args.enable_vae_tiling:          pipe.vae.enable_tiling()
            if args.xformers_efficient:             pipe.enable_xformers_memory_efficient_attention()
            if args.enable_sequential_cpu_offload:  pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
            elif args.enable_model_cpu_offload:     pipe.enable_model_cpu_offload(gpu_id=local_rank)
            else:                                   pipe = pipe.to(f"cuda:{local_rank}")

            # quantize
            if args.quantize_unet_to is not None:
                quantize_helper("transformer", pipe, args.quantize_unet_to, logger)
            if args.quantize_encoder_to is not None:
                quantize_helper("encoder", pipe, args.quantize_encoder_to, logger)
            if args.quantize_misc_to is not None:
                if hasattr(pipe, "controlnet") and pipe.controlnet is not None:
                    quantize_helper("manual", pipe, args.quantize_misc_to, logger, manual_module="controlnet")
                if hasattr(pipe, "motion_adapter") and pipe.motion_adapter is not None:
                    quantize_helper("manual", pipe, args.quantize_misc_to, logger, manual_module="motion_adapter")

            # set lora
            adapter_names = None
            if args.lora is not None:
                adapter_names = load_lora(args.lora, pipe, local_rank, logger, (args.quantize_to is not None))

            # compiles
            if args.compile_unet or args.compile_vae or args.compile_encoder:
                if args.compile_mode is not None and args.compile_options is not None:
                    logger.info("Compile mode and options are both defined, will ignore compile mode.")
                    args.compile_mode = None
                compiler_config                         = {}
                compiler_config["fullgraph"]            = (args.compile_fullgraph_off is None or args.compile_fullgraph_off == False)
                compiler_config["dynamic"]              = False
                if args.compile_backend is not None:    compiler_config["backend"] = args.compile_backend
                if args.compile_mode is not None:       compiler_config["mode"] = args.compile_mode
                if args.compile_options is not None:    compiler_config["options"] = json.loads(args.compile_options)
                if args.compile_unet:                   compile_helper("transformer", pipe, compiler_config, logger, adapter_names=adapter_names)
                if args.compile_vae:                    ompile_helper("vae", pipe, compiler_config, logger)
                if args.compile_encoder:                compile_helper("encoder", pipe, compiler_config, logger)

            # set progress bar visibility
            pipe.set_progress_bar_config(disable=local_rank != 0)

            # clean up
            clean()

            # warm up run
            if args.warm_up_steps is not None and args.warm_up_steps > 0:
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
    height,
    width,
    positive,
    negative,
    positive_embeds,
    negative_embeds,
    ip_image,
    latent,
    steps,
    seed,
    cfg,
    clip_skip,
    denoise,
    sigmas,
    timesteps,
    denoising_end,
):
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            global pipe, local_rank, input_config, result, step_progress
            args = get_args()
            device = torch.device("cuda", torch.cuda.current_device())
            torch_dtype = get_torch_type(args.variant)
            torch.cuda.reset_peak_memory_stats()
            step_progress = 0
            device = torch.device("cuda", torch.cuda.current_device())
            set_scheduler(args, pipe)

            if ip_image is not None and args.ip_adapter is not None:
                ip_image = load_image(ip_image)

            positive_pooled_embeds = None
            negative_pooled_embeds = None
            if positive_embeds is not None:
                positive_pooled_embeds = positive_embeds[0][1]["pooled_output"]
                positive_embeds = positive_embeds[0][0]
            if negative_embeds is not None:
                negative_pooled_embeds = negative_embeds[0][1]["pooled_output"]
                negative_embeds = negative_embeds[0][0]

            def set_step_progress(pipe, index, timestep, callback_kwargs):
                global get_torch_type, logger, process_input_latent, step_progress
                nonlocal args, denoise, device, latent, steps, torch_dtype
                # TODO: support xfuser-wrapped schedulers
                #the_index = get_scheduler_progressbar_offset_index(pipe.scheduler, index)
                step_progress = index / steps * 100
                if latent is not None:
                    if denoise is None or denoise > 1.0:
                        denoise = 1.0
                    target = int(steps * (1 - denoise))
                    if index == target:
                        latent = process_input_latent(latent, pipe, torch_dtype, device, timestep=timestep)
                        callback_kwargs["latents"] = latent
                        logger.info(f'Injected latent at step {target}')
                return callback_kwargs

            generator = torch.Generator(device="cpu").manual_seed(seed)

            is_image                                        = True
            kwargs                                          = {}
            kwargs["generator"]                             = generator
            kwargs["guidance_scale"]                        = cfg
            kwargs["num_inference_steps"]                   = steps
            kwargs["callback_on_step_end"]                  = set_step_progress
            kwargs["callback_on_step_end_tensor_inputs"]    = ["latents"]
            kwargs["width"]                                 = args.width
            kwargs["height"]                                = args.height
            kwargs["max_sequence_length"]                   = 256
            kwargs["output_type"]                           = "latent"
            kwargs["use_resolution_binning"]                = input_config.use_resolution_binning
            if height is not None:                          kwargs["height"]                    = height
            if width is not None:                           kwargs["width"]                     = width
            if positive is not None:                        kwargs["prompt"]                    = positive
            if negative is not None:                        kwargs["negative_prompt"]           = negative
            if positive_embeds is not None:                 kwargs["prompt_embeds"]             = positive_embeds
            if positive_pooled_embeds is not None:          kwargs["pooled_prompt_embeds"]      = positive_pooled_embeds
            if negative_embeds is not None:                 kwargs["negative_embeds"]           = negative_embeds
            if negative_pooled_embeds is not None:          kwargs["negative_pooled_embeds"]    = negative_pooled_embeds
            if latent is not None:                          kwargs["latents"]                   = latent
            if clip_skip is not None:                       kwargs["clip_skip"]                 = clip_skip
            if sigmas is not None:                          kwargs["sigmas"]                    = sigmas
            if timesteps is not None:                       kwargs["timesteps"]                 = timesteps
            if denoising_end is not None:                   kwargs["denoising_end"]             = denoising_end
            if args.ip_adapter is not None:
                if ip_image is not None:
                    kwargs["ip_adapter_image"] = ip_image
                else:
                    return "No IPAdapter image provided for a IPAdapter-loaded pipeline", None, False

            output = pipe(**kwargs)

            # clean up
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
                latents = output.images

                # from xdit process_latents(latents)
                latents = pipe._unpack_latents(latents, args.height, args.width, pipe.vae_scale_factor)
                # latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

                images = convert_latent_to_image(copy.copy(latents.to(device)), pipe)
                latents = convert_latent_to_output_latent(copy.copy(latents.to(device)), pipe)
                with app.app_context():
                    requests.post(f"http://localhost:{args.port}/set_result", json={ "image": pickle_and_encode_b64(images[0]), "latent": pickle_and_encode_b64(latents) })


@app.route("/set_result", methods=["POST"])
def set_result():
    global result
    result = request.json
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
    latent              = data.get("latent")
    steps               = data.get("steps")
    seed                = data.get("seed")
    cfg                 = data.get("cfg")
    clip_skip           = data.get("clip_skip")
    denoise             = data.get("denoise")
    sigmas              = data.get("sigmas")
    timesteps           = data.get("timesteps")
    denoising_end       = data.get("denoising_end")
    height              = data.get("height")
    width               = data.get("width")

    print_params(data, logger)

    if height is None:                                          height = args.height
    if width is None:                                           width = args.width
    if positive is not None and len(positive) == 0:             positive = None
    if negative is not None and len(negative) == 0:             negative = None
    if positive is None and positive_embeds is None:            jsonify({ "message": "No input provided", "output": None, "is_image": False })
    if positive is not None and positive_embeds is not None:    jsonify({ "message": "Provide only one positive input", "output": None, "is_image": False })
    if negative is not None and negative_embeds is not None:    jsonify({ "message": "Provide only one negative input", "output": None, "is_image": False })
    if ip_image is not None:                                    ip_image = decode_b64_and_unpickle(ip_image)
    if latent is not None:                                      latent = decode_b64_and_unpickle(latent)
    if positive_embeds is not None:                             positive_embeds = decode_b64_and_unpickle(positive_embeds)
    if negative_embeds is not None:                             negative_embeds = decode_b64_and_unpickle(negative_embeds)
    if sigmas is not None:                                      sigmas = decode_b64_and_unpickle(sigmas)
    if timesteps is not None:                                   timesteps = decode_b64_and_unpickle(timesteps)

    params = [
        dummy,
        height,
        width,
        positive,
        negative,
        positive_embeds,
        negative_embeds,
        ip_image,
        latent,
        steps,
        seed,
        cfg,
        clip_skip,
        denoise,
        sigmas,
        timesteps,
        denoising_end,
    ]
    dist.broadcast_object_list(params, src=0)
    message, outputs, is_image = generate_image_parallel(*params)
    response = { "message": message, "output": outputs["image"], "latent": outputs["latent"], "is_image": is_image }
    return jsonify(response)


def run_host():
    global logger
    args = get_args()
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        app.run(host="localhost", port=args.port)
    else:
        while True:
            params = [None] * 17 # len(params) of generate_image_parallel()
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
