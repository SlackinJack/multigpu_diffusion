import argparse
import base64
import copy
import gc
import json
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from compel import Compel, ReturnedEmbeddingsType
from DeepCache import DeepCacheSDHelper
from diffusers import (
    AnimateDiffControlNetPipeline,
    AnimateDiffPipeline,
    AutoencoderKL,
    AutoencoderKLTemporalDecoder,
    ControlNetModel,
    MotionAdapter,
    QuantoConfig,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
    UNet2DConditionModel,
    UNetSpatioTemporalConditionModel,
)
from diffusers.utils import load_image
from flask import Flask, request, jsonify
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPVisionModelWithProjection


from AsyncDiff.asyncdiff.async_animate import AsyncDiff as AsyncDiffAD
from AsyncDiff.asyncdiff.async_sd import AsyncDiff as AsyncDiffSD


from modules.host_generics import *
from modules.scheduler_config import *
from modules.utils import *


app = Flask(__name__)
async_diff = None
initialized = False
step_progress = 0
local_rank = None
logger = None
pipe = None
can_cache = False
cache_kwargs = { "cache_interval": 3, "cache_branch_id": 0 }


def get_args():
    parser = argparse.ArgumentParser()
    # asyncdiff
    parser.add_argument("--model_n", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--stride", type=int, default=1, choices=[1, 2])
    parser.add_argument("--warm_up", type=int, default=3)
    parser.add_argument("--time_shift", action="store_true")
    # generic
    for k, v in GENERIC_HOST_ARGS.items():  parser.add_argument(f"--{k}", type=v, default=None)
    for e in GENERIC_HOST_ARGS_TOGGLES:     parser.add_argument(f"--{e}", action="store_true")
    args = parser.parse_args()
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
    global pipe, local_rank, async_diff, initialized, can_cache, cache_kwargs, logger
    args = get_args()

    # checks
    assert not (args.type == "ad" and args.motion_adapter is None and args.motion_module is None), "AnimateDiff requires providing a motion adapter/module."
    # assert (args.ip_adapter is None or args.quantize_to is None), "IPAdapter is not supported when using quantization"
    if args.type in ["ad", "sd3", "sdup"]: can_cache = False

    # init distributed inference
    mp.set_start_method("spawn", force=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    logger = get_logger(local_rank)
    # dynamo tweaks
    setup_torch_dynamo()
    # torch tweaks
    setup_torch_backends()

    # set torch type
    torch_dtype = get_torch_type(args.variant)

    # quantize
    quant = {}
    if args.quantize_to is not None:
        quant = {"quantization_config": QuantoConfig(weights_dtype=args.quantize_to)}
    def quantize(model, desc):
        do_quantization(model, desc, args.quantize_to, logger)
        return

    # set pipeline
    logger.info(f"Initializing pipeline")
    kwargs = {}
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

    # set control net
    controlnet_model = None
    if args.control_net is not None:
        args.control_net = json.loads(args.control_net)
        for k, v in args.control_net.items():
            controlnet_model = ControlNetModel.from_pretrained(k, **quant, **kwargs_model)
            break
        if args.quantize_to:    to_quantize["controlnet"] = controlnet_model
        else:                   kwargs["controlnet"] = controlnet_model

    match args.type:
        case "ad":
            if args.motion_module is not None:
                adapter = MotionAdapter.from_single_file(
                    args.motion_module,
                    config=f"{os.path.dirname(__file__)}/resources/generic_motion_adapter_config.json",
                    torch_dtype=torch_dtype,
                    use_safetensors=False, # NOTE: safetensors off
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    **quant
                )
            else:
                adapter = MotionAdapter.from_pretrained(args.motion_adapter, **quant, **kwargs_model)
            if args.quantize_to is not None:    to_quantize["motion_adapter"] = adapter
            else:                               kwargs["motion_adapter"] = adapter
            PipelineClass = AnimateDiffControlNetPipeline if args.control_net is not None else AnimateDiffPipeline
        case "sd1":
            if args.quantize_to is not None:
                to_quantize["text_encoder"] = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder", **kwargs_model)
                if args.ip_adapter is None:
                    to_quantize["unet"] = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet", **quant, **kwargs_model).to(f'cuda:{local_rank}')
                else:
                    quantize_unet_after = True
            kwargs["vae"] = AutoencoderKL.from_pretrained(args.model, subfolder="vae", **kwargs_model)
            PipelineClass = StableDiffusionControlNetPipeline if args.control_net is not None else StableDiffusionPipeline
        case "sd2":
            if args.quantize_to is not None:
                to_quantize["text_encoder"] = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder", **kwargs_model)
                if args.ip_adapter is None:
                    to_quantize["unet"] = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet", **quant, **kwargs_model).to(f'cuda:{local_rank}')
                else:
                    quantize_unet_after = True
            kwargs["vae"] = AutoencoderKL.from_pretrained(args.model, subfolder="vae", **kwargs_model)
            PipelineClass = StableDiffusionControlNetPipeline if args.control_net is not None else StableDiffusionPipeline
        case "sd3":
            if args.quantize_to is not None:
                to_quantize["text_encoder"] = CLIPTextModelWithProjection.from_pretrained(args.model, subfolder="text_encoder", **kwargs_model)
                to_quantize["text_encoder_2"] = CLIPTextModelWithProjection.from_pretrained(args.model, subfolder="text_encoder_2", **kwargs_model)
                to_quantize["text_encoder_3"] = T5EncoderModel.from_pretrained(args.model, subfolder="text_encoder_3", **kwargs_model)
                if args.ip_adapter is None:
                    to_quantize["transformer"] = SD3Transformer2DModel.from_pretrained(args.model, subfolder="transformer", **quant, **kwargs_model).to(f'cuda:{local_rank}')
                else:
                    quantize_unet_after = True
            kwargs["vae"] = AutoencoderKL.from_pretrained(args.model, subfolder="vae", **kwargs_model)
            PipelineClass = StableDiffusion3ControlNetPipeline if args.control_net is not None else StableDiffusion3Pipeline
        case "sdup":
            if args.quantize_to is not None:
                to_quantize["text_encoder"] = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder", **kwargs_model)
                to_quantize["unet"] = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet", **quant, **kwargs_model).to(f'cuda:{local_rank}')
            kwargs["vae"] = AutoencoderKL.from_pretrained(args.model, subfolder="vae", **kwargs_model)
            PipelineClass = StableDiffusionUpscalePipeline
        case "sdxl":
            if args.quantize_to is not None:
                to_quantize["text_encoder"] = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder", **kwargs_model)
                to_quantize["text_encoder_2"] = CLIPTextModelWithProjection.from_pretrained(args.model, subfolder="text_encoder_2", **kwargs_model)
                if args.ip_adapter is None:
                    to_quantize["unet"] = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet", **quant, **kwargs_model).to(f'cuda:{local_rank}')
                else:
                    quantize_unet_after = True
            kwargs["vae"] = AutoencoderKL.from_pretrained(args.model, subfolder="vae", **kwargs_model)
            PipelineClass = StableDiffusionXLControlNetPipeline if args.control_net is not None else StableDiffusionXLPipeline
        case "svd":
            if args.quantize_to is not None:
                to_quantize["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(args.model, subfolder="image_encoder", **kwargs_model)
                to_quantize["unet"] = UNetSpatioTemporalConditionModel.from_pretrained(args.model, subfolder="unet", **quant, **kwargs_model).to(f'cuda:{local_rank}')
            kwargs["vae"] = AutoencoderKLTemporalDecoder.from_pretrained(args.model, subfolder="vae", **kwargs_model)
            PipelineClass = StableVideoDiffusionPipeline
        case _: raise NotImplementedError

    # set vae
    if args.vae is not None:
        vae = AutoencoderKL.from_pretrained(args.vae, **kwargs_model)
        kwargs["vae"] = vae

    if len(to_quantize) > 0:
        for k, v in to_quantize.items():
            quantize(v, k)
            kwargs[k] = v

    pipe = PipelineClass.from_pretrained(args.model, **quant, **kwargs)
    logger.info(f"Pipeline initialized")

    # set ipadapter
    if args.ip_adapter is not None:
        args.ip_adapter = json.loads(args.ip_adapter)
        for m, s in args.ip_adapter.items():
            split = m.split("/")
            ip_adapter_file = split[-1]
            ip_adapter_subfolder = split[-2]
            ip_adapter_folder = m.replace(f'/{ip_adapter_subfolder}/{ip_adapter_file}', "")
            pipe.load_ip_adapter(
                ip_adapter_folder,
                subfolder=ip_adapter_subfolder,
                weight_name=ip_adapter_file,
                torch_dtype=torch_dtype,
                use_safetensors=False, # NOTE: safetensors off
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            pipe.set_ip_adapter_scale(s)

    # deferred quantize
    if quantize_unet_after:
        if args.type in ["sd3"]:
            quantize(pipe.transformer, "transformer")
        else:
            quantize(pipe.unet, "unet")

    # set lora
    adapter_names = None
    if args.lora is not None and args.type in ["sd1", "sd2", "sd3", "sdxl"]:
        adapter_names = load_lora(args.lora, pipe, local_rank, logger, (args.quantize_to is not None))
        if len(to_quantize) > 0:
            logger.info("Requantizing unet/transformer, text encoder(s)")
            if to_quantize.get("unet") is not None:
                quantize(pipe.unet, "unet")
            if to_quantize.get("transformer") is not None:
                quantize(pipe.transformer, "transformer")
            if to_quantize.get("text_encoder") is not None:
                quantize(pipe.text_encoder, "text_encoder")
            if to_quantize.get("text_encoder_2") is not None:
                quantize(pipe.text_encoder_2, "text_encoder_2")
            if to_quantize.get("text_encoder_3") is not None:
                quantize(pipe.text_encoder_3, "text_encoder_3")

    # set scheduler
    if args.scheduler is not None and args.type in ["ad", "sd1", "sd2", "sd3", "sdup", "sdxl"]:
        pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)

    # set memory saving
    if args.type not in ["svd"]:
        if args.enable_vae_slicing: pipe.vae.enable_slicing()
        if args.enable_vae_tiling:  pipe.vae.enable_tiling()
        if args.xformers_efficient: pipe.enable_xformers_memory_efficient_attention()
    if args.enable_sequential_cpu_offload: logger.info("sequential CPU offload not supported - ignoring")
    if args.enable_model_cpu_offload: logger.info("model CPU offload not supported - ignoring")

    # compiles
    if args.compile_unet:
        if args.type in ["sd3"]:                                compile_transformer(pipe, adapter_names, logger)
        else:                                                   compile_unet(pipe, adapter_names, logger)
    if args.compile_vae:                                        compile_vae(pipe, logger)
    if args.type not in ["svd"] and args.compile_text_encoder:  compile_text_encoder(pipe, logger)

    # set asyncdiff
    if args.type in ["ad"]: ad_class = AsyncDiffAD
    else:                   ad_class = AsyncDiffSD
    async_diff = ad_class(pipe, args.type, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)

    # set progress bar visibility
    pipe.set_progress_bar_config(disable=dist.get_rank() != 0)

    # clean up
    clean()

    # warm up run
    if args.warm_up_steps > 0:
        def get_warmup_image():
            # image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
            image = load_image(f"{os.path.dirname(__file__)}/resources/rocket.png") # 1024x576 pixels
            image = image.resize((768, 432), Image.Resampling.LANCZOS)
            return image
        generator = torch.Generator(device="cpu").manual_seed(1)
        async_diff.reset_state(warm_up=args.warm_up)

        prompt = "a dog"
        cfg = 7
        frames = 25
        chunk_size = 8

        kwargs = {}
        kwargs["width"] = args.width
        kwargs["height"] = args.height
        kwargs["num_inference_steps"] = args.warm_up_steps
        match args.type:
            case "ad":
                if args.ip_adapter is not None:
                    kwargs["ip_adapter_image"] = get_warmup_image()
                    if args.control_net is not None:
                        kwargs["conditioning_frames"] = [get_warmup_image()] * frames
                kwargs["prompt"] = prompt
                kwargs["num_frames"] = frames
                kwargs["guidance_scale"] = cfg
            case "sdup":
                kwargs["prompt"] = prompt
                kwargs["image"] = get_warmup_image()
            case "svd":
                kwargs["image"] = get_warmup_image()
                kwargs["decode_chunk_size"] = chunk_size
            case _:
                kwargs["prompt"] = prompt
                if args.ip_adapter is not None:
                    kwargs["ip_adapter_image"] = get_warmup_image()
                if args.control_net is not None:
                    for k, v in args.control_net.items():
                        kwargs["image"] = get_warmup_image()
                        kwargs["controlnet_conditioning_scale"] = v
        logger.info("Starting warmup run")
        if can_cache:
            helper = DeepCacheSDHelper(pipe=pipe)
            helper.set_params(**cache_kwargs)
            helper.enable()
        pipe(**kwargs)
        if can_cache:
            helper.disable()
            del helper

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
    image,
    image_scale,
    ip_image,
    ip_image_scale,
    control_image,
    control_image_scale,
    latent,
    steps,
    cfg,
    controlnet_scale,
    seed,
    frames,
    decode_chunk_size,
    clip_skip,
    motion_bucket_id,
    noise_aug_strength
):
    global async_diff, pipe, step_progress, can_cache, cache_kwargs
    args = get_args()
    torch.cuda.reset_peak_memory_stats()
    async_diff.reset_state(warm_up=args.warm_up)
    step_progress = 0

    if args.type in ["sdup", "svd"]:
        if image is None: return "No image provided for an image pipeline.", None, False
        image = load_image(image)
        if image_scale is not None and image_scale != 100:
            percentage = image_scale / 100
            image = image.resize((int(image.size[0] * percentage), int(image.size[1] * percentage)), Image.Resampling.LANCZOS)
    if ip_image is not None and args.ip_adapter is not None:
        ip_image = load_image(ip_image)
        if ip_image_scale is not None and ip_image_scale != 100:
            percentage = ip_image_scale / 100
            ip_image = ip_image.resize((int(ip_image.size[0] * percentage), int(ip_image.size[1] * percentage)), Image.Resampling.LANCZOS)
    if control_image is not None and args.control_net is not None:
        control_image = load_image(control_image)
        if control_image_scale is not None and control_image_scale != 100:
            percentage = control_image_scale / 100
            control_image = control_image.resize((int(control_image.size[0] * percentage), int(control_image.size[1] * percentage)), Image.Resampling.LANCZOS)

    if latent is not None:
        latent = latent.to(get_torch_type(args.variant))

    def set_step_progress(pipe, index, timestep, callback_kwargs):
        global step_progress
        step_progress = index / steps * 100
        return callback_kwargs

    generator = torch.Generator(device="cpu").manual_seed(seed)

    can_use_compel = args.compel and not positive_embeds and not negative_embeds and args.type in ["sd1", "sd2", "sdxl"]
    if can_use_compel:
        if args.type in ["sd1", "sd2"]:
            embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
        else:
            embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=embeddings_type,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
        positive_embeds, positive_pooled_embeds = compel([positive])
        if negative is not None and len(negative) > 0:
            negative_embeds, negative_pooled_embeds = compel([negative])
        positive = None
        negative = None

    kwargs = {}
    kwargs["generator"] = generator
    kwargs["num_inference_steps"] = steps
    kwargs["callback_on_step_end"] = set_step_progress
    match args.type:
        case "ad":
            is_image = False
            if ip_image is not None:
                kwargs["ip_adapter_image"] = ip_image
            if args.control_net is not None:
                kwargs["conditioning_frames"] = [control_image] * frames
            if positive is not None:    kwargs["prompt"] = positive
            if negative is not None:    kwargs["negative_prompt"] = negative
            kwargs["width"] = args.width
            kwargs["height"] = args.height
            kwargs["num_frames"] = frames
            kwargs["guidance_scale"] = cfg
            kwargs["output_type"] = "pil"
        case "sdup":
            is_image = True
            if positive is not None:    kwargs["prompt"] = positive
            if negative is not None:    kwargs["negative_prompt"] = negative
            if image is not None:       kwargs["image"] = image
            kwargs["guidance_scale"] = cfg
            kwargs["output_type"] = "pil"
        case "svd":
            is_image = False
            if image is not None: kwargs["image"] = image
            kwargs["width"] = args.width
            kwargs["height"] = args.height
            kwargs["num_frames"] = frames
            kwargs["decode_chunk_size"] = decode_chunk_size
            kwargs["motion_bucket_id"] = motion_bucket_id
            kwargs["noise_aug_strength"] = noise_aug_strength
            kwargs["output_type"] = "pil"
        case _:
            is_image = True
            positive_pooled_embeds = None
            negative_pooled_embeds = None
            if positive_embeds is not None:
                positive_pooled_embeds = positive_embeds[0][1]["pooled_output"]
                positive_embeds = positive_embeds[0][0]
            if negative_embeds is not None:
                negative_pooled_embeds = negative_embeds[0][1]["pooled_output"]
                negative_embeds = negative_embeds[0][0]

            if positive is not None:                                    kwargs["prompt"]                    = positive
            if negative is not None:                                    kwargs["negative_prompt"]           = negative
            if positive_embeds is not None:                             kwargs["prompt_embeds"]             = positive_embeds
            if positive_pooled_embeds is not None:                      kwargs["pooled_prompt_embeds"]      = positive_pooled_embeds
            if negative_embeds is not None:                             kwargs["negative_embeds"]           = negative_embeds
            if negative_pooled_embeds is not None:                      kwargs["negative_pooled_embeds"]    = negative_pooled_embeds
            if latent is not None:                                      kwargs["latents"]                   = latent
            if args.ip_adapter is not None:
                if ip_image is not None:
                    kwargs["ip_adapter_image"] = ip_image
                else:
                    return "No IPAdapter image provided for a IPAdapter-loaded pipeline", None, False
            if args.control_net is not None:
                if control_image is not None:
                    for k, v in json.loads(args.control_net).items():
                        kwargs["image"] = control_image
                        kwargs["controlnet_conditioning_scale"] = v
                else:
                    return "No ConstrolNet image provided for a ControlNet-loaded pipeline", None, False
            kwargs["width"] = args.width
            kwargs["height"] = args.height
            kwargs["clip_skip"] = clip_skip
            kwargs["guidance_scale"] = cfg
            kwargs["output_type"] = "pil"

    if can_cache:
        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(**cache_kwargs)
        helper.enable()
    output = pipe(**kwargs)
    if can_cache:
        helper.disable()
        del helper

    if can_use_compel:
        # https://github.com/damian0815/compel/issues/24
        positive_embeds = positive_pooled_embeds = negative_embeds = negative_pooled_embeds = None

    # clean up
    clean()

    if dist.get_rank() == 0:
        step_progress = 100
        if output is not None:
            if is_image:    output = output.images[0]
            else:           output = output.frames[0]
            return "OK", pickle_and_encode_b64(output), is_image
        else:
            return "No image from pipeline", None, False


@app.route("/generate", methods=["POST"])
def generate_image():
    global logger
    args = get_args()
    data = request.json
    dummy               = 0
    positive            = data.get("positive")
    negative            = data.get("negative")
    positive_embeds     = data.get("positive_embeds")
    negative_embeds     = data.get("negative_embeds")
    image               = data.get("image")
    image_scale         = data.get("image_scale")
    ip_image            = data.get("ip_image")
    ip_image_scale      = data.get("ip_image_scale")
    control_image       = data.get("control_image")
    control_image_scale = data.get("control_image_scale")
    latent              = data.get("latent")
    steps               = data.get("steps")
    cfg                 = data.get("cfg")
    controlnet_scale    = data.get("controlnet_scale")
    seed                = data.get("seed")
    frames              = data.get("frames")
    decode_chunk_size   = data.get("decode_chunk_size")
    clip_skip           = data.get("clip_skip")
    motion_bucket_id    = data.get("motion_bucket_id")
    noise_aug_strength  = data.get("noise_aug_strength")

    if positive is not None and len(positive) == 0:                     positive = None
    if negative is not None and len(negative) == 0:                     negative = None
    if image is None and positive is None and positive_embeds is None:  jsonify({ "message": "No input provided", "output": None, "is_image": False })
    if positive is not None and positive_embeds is not None:            jsonify({ "message": "Provide only one positive input", "output": None, "is_image": False })
    if negative is not None and negative_embeds is not None:            jsonify({ "message": "Provide only one negative input", "output": None, "is_image": False })
    if image is not None:                                               image = decode_b64_and_unpickle(image)
    if ip_image is not None:                                            ip_image = decode_b64_and_unpickle(ip_image)
    if control_image is not None:                                       control_image = decode_b64_and_unpickle(control_image)
    if latent is not None:                                              latent = decode_b64_and_unpickle(latent)
    if positive_embeds is not None:                                     positive_embeds = decode_b64_and_unpickle(positive_embeds)
    if negative_embeds is not None:                                     negative_embeds = decode_b64_and_unpickle(negative_embeds)

    params = [
        dummy,
        positive,
        negative,
        positive_embeds,
        negative_embeds,
        image,
        image_scale,
        ip_image,
        ip_image_scale,
        control_image,
        control_image_scale,
        latent,
        steps,
        cfg,
        controlnet_scale,
        seed,
        frames,
        decode_chunk_size,
        clip_skip,
        motion_bucket_id,
        noise_aug_strength
    ]
    dist.broadcast_object_list(params, src=0)
    print_params(
        {
            "positive": positive,
            "negative": negative,
            "positive_embeds": (positive_embeds is not None),
            "negative_embeds": (negative_embeds is not None),
            "image": (image is not None),
            "image_scale": image_scale,
            "ip_image": (ip_image is not None),
            "ip_image_scale": ip_image_scale,
            "control_image": (control_image is not None),
            "control_image_scale": control_image_scale,
            "latent": (latent is not None),
            "steps": steps,
            "cfg": cfg,
            "controlnet_scale": controlnet_scale,
            "seed": seed,
            "frames": frames,
            "decode_chunk_size": decode_chunk_size,
            "clip_skip": clip_skip,
            "motion_bucket_id": motion_bucket_id,
            "noise_aug_strength": noise_aug_strength,
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
            params = [None] * 21 # len(params) of generate_image_parallel()
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
