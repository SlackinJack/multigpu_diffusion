import argparse
import base64
import copy
import json
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    AnimateDiffControlNetPipeline,
    AnimateDiffPipeline,
    FluxControlNetPipeline,
    FluxPipeline,
    QuantoConfig,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
    ZImagePipeline,
    # ZImageControlNetPipeline,
)
from diffusers.utils import load_image
from flask import Flask, request, jsonify
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPVisionModelWithProjection


from AsyncDiff.asyncdiff.async_animate import AsyncDiff as AsyncDiffAD
from AsyncDiff.asyncdiff.async_flux import AsyncDiff as AsyncDiffF
from AsyncDiff.asyncdiff.async_sd import AsyncDiff as AsyncDiffSD
from AsyncDiff.asyncdiff.async_sd3 import AsyncDiff as AsyncDiffSD3
from AsyncDiff.asyncdiff.async_zimage import AsyncDiff as AsyncDiffZ


from modules.host_generics import *
from modules.scheduler_config import *
from modules.utils import *


app = Flask(__name__)
async_diff = None
local_rank = None


def get_args():
    parser = argparse.ArgumentParser()
    # asyncdiff
    parser.add_argument("--model_n",        type=int,   default=2) # NOTE: if n > 4, you'll need to manually map your model in pipe_config.py
    parser.add_argument("--stride",         type=int,   default=1)
    parser.add_argument("--synced_steps",   type=int,   default=3)
    parser.add_argument("--synced_percent", type=float, default=0.0)
    parser.add_argument("--time_shift",     action="store_true")
    # generic
    for k, v in GENERIC_HOST_ARGS.items():  parser.add_argument(f"--{k}", type=v, default=None)
    for e in GENERIC_HOST_ARGS_TOGGLES:     parser.add_argument(f"--{e}", action="store_true")
    args = parser.parse_args()
    return args


@app.route("/<path>", methods=["GET", "POST"])
def handle_path(path):
    match path:
        case "initialize":
            return get_initialized_flask()
        case "progress":
            return get_progress_flask()
        case "generate":
            return generate_image()
        case _:
            return "", 404


def initialize():
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            global local_rank, async_diff
            args = get_args()

            # checks
            assert not (args.type == "ad" and args.motion_adapter is None and args.motion_module is None), "AnimateDiff requires providing a motion adapter/module."

            # init distributed inference
            mp.set_start_method("spawn", force=True)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            set_logger(local_rank)

            # dynamo tweaks
            setup_torch_dynamo(args.torch_cache_limit, args.torch_accumlated_cache_limit, args.torch_capture_scalar)

            # torch tweaks
            setup_torch_backends()

            # set torch type
            torch_dtype = get_torch_type(args.variant)

            # set pipeline
            get_logger().info(f"Initializing pipeline")
            kwargs = {}
            kwargs["torch_dtype"] = torch_dtype
            kwargs["use_safetensors"] = True
            kwargs["local_files_only"] = True
            kwargs["low_cpu_mem_usage"] = True
            kwargs["add_watermarker"] = False

            # quantize
            is_quantized = False
            mappings = {}
            if args.quantize_unet is not None:
                mappings.update(get_quant_mapping("unet", args.quantize_unet))
            if args.quantize_encoder is not None:
                mappings.update(get_quant_mapping("encoder", args.quantize_encoder))
            if args.quantize_vae is not None:
                mappings.update(get_quant_mapping("vae", args.quantize_vae))
            if args.quantize_tokenizer is not None:
                mappings.update(get_quant_mapping("tokenizer", args.quantize_tokenizer))
            if args.quantize_scheduler is not None:
                mappings.update(get_quant_mapping("scheduler", args.quantize_scheduler))
            if args.quantize_misc is not None:
                mappings.update(get_quant_mapping("misc", args.quantize_misc))
            if len(list(mappings.keys())) > 0:
                is_quantized = True
                kwargs["quantization_config"] = get_pipe_quant_config(mappings)

            PipelineClass = None

            # set control net
            controlnet_model = None
            if args.control_net is not None and args.control_net_config is not None and args.type not in ["sdup", "svd"]:
                args.control_net = json.loads(args.control_net)
                assert len(args.control_net) == 1, "Multiple ControlNets are not current supported."
                k, v = list(args.control_net.items())[0]
                controlnet_model = load_model(k, args.control_net_config, "ControlNetModel", torch_dtype)
                kwargs["controlnet"] = controlnet_model

            # set unet
            if args.unet is not None and args.unet_config is not None:
                kwargs["unet"] = load_model(args.unet, args.unet_config, "UNet2DConditionModel", torch_dtype)

            # set transformer
            if args.transformer is not None and args.transformer_config is not None:
                match args.type:
                    case "flux":
                        kwargs["transformer"] = load_model(args.transformer, args.transformer_config, "FluxTransformer2DModel", torch_dtype)
                    case "sd3":
                        kwargs["transformer"] = load_model(args.transformer, args.transformer_config, "SD3Transformer2DModel", torch_dtype)
                    case "zimage":
                        kwargs["transformer"] = load_model(args.transformer, args.transformer_config, "ZImageTransformer2DModel", torch_dtype)

            # set vae
            if args.vae is not None and args.vae_config is not None and args.type not in ["ad", "svd"]:
                kwargs["vae"] = load_model(args.vae, args.vae_config, "AutoencoderKL", torch_dtype)

            # set motion_adapter
            if (args.motion_module is not None or args.motion_adapter is not None) and args.motion_config is not None and args.type in ["ad"]:
                if args.motion_module is not None:
                    kwargs["motion_adapter"] = load_model(args.motion_module, args.motion_config, "MotionAdapter", torch_dtype)
                else:
                    kwargs["motion_adapter"] = load_model(args.motion_adapter, args.motion_config, "MotionAdapter", torch_dtype)

            match args.type:
                case "ad":
                    PipelineClass = AnimateDiffControlNetPipeline if args.control_net is not None else AnimateDiffPipeline
                case "flux":
                    PipelineClass = FluxControlNetPipeline if args.control_net is not None else FluxPipeline
                case "sd1":
                    PipelineClass = StableDiffusionControlNetPipeline if args.control_net is not None else StableDiffusionPipeline
                case "sd2":
                    PipelineClass = StableDiffusionControlNetPipeline if args.control_net is not None else StableDiffusionPipeline
                case "sd3":
                    PipelineClass = StableDiffusion3ControlNetPipeline if args.control_net is not None else StableDiffusion3Pipeline
                case "sdup":
                    PipelineClass = StableDiffusionUpscalePipeline
                case "sdxl":
                    PipelineClass = StableDiffusionXLControlNetPipeline if args.control_net is not None else StableDiffusionXLPipeline
                case "svd":
                    PipelineClass = StableVideoDiffusionPipeline
                case "zimage":
                    # PipelineClass = ZImageControlNetPipeline if args.control_net is not None else ZImagePipeline
                    PipelineClass = ZImagePipeline
                case _: raise NotImplementedError

            # init pipe
            set_pipe(PipelineClass.from_pretrained(args.checkpoint, **kwargs))
            del kwargs
            get_logger().info("Pipeline initialized")

            # for debug - to print model
            if local_rank == 0:
                # get_logger().info(str(get_pipe().transformer))
                # raise ValueError
                pass

            # set scheduler
            set_scheduler(args)

            # set ipadapter
            if args.ip_adapter is not None:
                args.ip_adapter = json.loads(args.ip_adapter)
                load_ip_adapter(args.ip_adapter)

            # set memory saving
            if args.type not in ["svd"]:
                if args.enable_vae_slicing:         get_pipe().vae.enable_slicing()
                if args.enable_vae_tiling:          get_pipe().vae.enable_tiling()
                if args.type not in ["flux"]:
                    if args.xformers_efficient:     get_pipe().enable_xformers_memory_efficient_attention()
            if args.enable_sequential_cpu_offload:  get_logger().info("sequential CPU offload not supported - ignoring")
            if args.enable_model_cpu_offload:       get_logger().info("model CPU offload not supported - ignoring")

            # set lora
            adapter_names = None
            if args.lora is not None and args.type not in ["ad", "sdup", "svd"]:
                adapter_names = load_lora(args.lora, local_rank)

            # compiles
            if args.compile_unet or args.compile_vae or args.compile_encoder:
                if args.compile_mode is not None and args.compile_options is not None:
                    get_logger().info("Compile mode and options are both defined, will ignore compile mode.")
                    args.compile_mode = None
                compiler_config                                 = {}
                compiler_config["fullgraph"]                    = (args.compile_fullgraph_off is None or args.compile_fullgraph_off == False)
                compiler_config["dynamic"]                      = False
                if args.compile_backend is not None:            compiler_config["backend"] = args.compile_backend
                if args.compile_mode is not None:               compiler_config["mode"] = args.compile_mode
                if args.compile_options is not None:            compiler_config["options"] = json.loads(args.compile_options)

                if args.compile_unet:
                    if args.type in ["flux", "sd3", "zimage"]:  compile_helper("transformer", compiler_config, adapter_names=adapter_names)
                    else:                                       compile_helper("unet", compiler_config, adapter_names=adapter_names)
                if args.compile_vae:                            compile_helper("vae", compiler_config)
                if args.compile_encoder:                        compile_helper("encoder", compiler_config)

            # set asyncdiff
            if args.type in ["ad"]:
                ad_class = AsyncDiffAD
            elif args.type in ["flux"]:
                ad_class = AsyncDiffF
            elif args.type in ["sd3"]:
                ad_class = AsyncDiffSD3
            elif args.type in ["zimage"]:
                ad_class = AsyncDiffZ
            else:
                ad_class = AsyncDiffSD
            async_diff = ad_class(get_pipe(), args.type, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)

            # set progress bar visibility
            get_pipe().set_progress_bar_config(disable=dist.get_rank() != 0)

            # set models to eval mode
            setup_evals()

            # clean up
            clean()

            # warm up run
            if args.warm_up_steps is not None and args.warm_up_steps > 0:
                generator = torch.Generator(device="cpu").manual_seed(1)
                if args.synced_percent is not None:
                    async_diff.reset_state(warm_up=(args.warm_up_steps * args.synced_percent) // 100)
                else:
                    async_diff.reset_state(warm_up=args.synced_steps)

                prompt      = "a dog"
                cfg         = 7
                frames      = 25
                chunk_size  = 8

                kwargs                          = {}
                kwargs["width"]                 = args.width
                kwargs["height"]                = args.height
                kwargs["num_inference_steps"]   = args.warm_up_steps
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

                get_logger().info("Starting warmup run")
                get_pipe().vae = get_pipe().vae.to(torch_dtype)
                get_pipe()(**kwargs)
                get_pipe().vae = get_pipe().vae.to(torch.float32)

            # clean up
            clean()

            # complete
            get_logger().info("Model initialization completed")
            if dist.get_rank() == 0:
                print_mem_usage()
            set_initialized(True)
            return


def generate_image_parallel(
    dummy,
    height,
    width,
    positive,
    negative,
    positive_embeds,
    negative_embeds,
    image,
    ip_image,
    control_image,
    latent,
    steps,
    cfg,
    controlnet_scale,
    seed,
    frames,
    decode_chunk_size,
    clip_skip,
    motion_bucket_id,
    noise_aug_strength,
    sigmas,
    timesteps,
    denoising_start,
    denoising_end,
):

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            global async_diff
            args = get_args()
            device = torch.device("cuda", torch.cuda.current_device())
            torch_dtype = get_torch_type(args.variant)
            torch.cuda.reset_peak_memory_stats()
            if args.synced_percent is not None:
                async_diff.reset_state(warm_up=(steps * args.synced_percent) // 100)
            else:
                async_diff.reset_state(warm_up=args.synced_steps)
            set_progress(0)
            set_scheduler(args)

            # checks
            if args.type in ["sdup", "svd"] and image is None:
                return "No image provided for an image pipeline.", None, False
            if sigmas is not None and timesteps is not None:
                return "Either sigmas or timesteps can be defined, but not both.", None, False
            if args.ip_adapter is not None and ip_image is None:
                return "No IPAdapter image provided for a IPAdapter-loaded pipeline", None, False
            if args.control_net is not None and control_image is None:
                return "No ConstrolNet image provided for a ControlNet-loaded pipeline", None, False

            # load image
            if args.type in ["sdup", "svd"]:
                image = load_image(image)
            if ip_image is not None and args.ip_adapter is not None:
                ip_image = load_image(ip_image)
            if control_image is not None and args.control_net is not None:
                control_image = load_image(control_image)

            # progress bar
            def set_step_progress(pipe, index, timestep, callback_kwargs):
                global get_torch_type, process_input_latent
                nonlocal args, denoising_start, device, latent, steps, torch_dtype
                the_index = get_scheduler_progressbar_offset_index(pipe.scheduler, index)
                set_progress(the_index / steps * 100)
                if latent is not None:
                    if denoising_start is None or denoising_start > 1.0:
                        denoising_start = 1.0
                    target = int(steps * (1 - denoising_start))
                    if the_index == target:
                        latent = process_input_latent(latent, torch_dtype, device, timestep=timestep)
                        callback_kwargs["latents"] = latent.to(device=pipe.unet.device, dtype=pipe.unet.dtype)
                        get_logger().info(f'Injected latent at step {target}/{steps}')
                return callback_kwargs

            # set seed
            generator = torch.Generator(device="cpu").manual_seed(seed)

            # compel
            positive_pooled_embeds = None
            negative_pooled_embeds = None
            if args.compel and args.type in ["sd1", "sd2", "sdxl"] and positive_embeds is None and negative_embeds is None:
                if args.type in ["sd1", "sd2"]: embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
                else:                           embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
                compel = Compel(
                    tokenizer=[get_pipe().tokenizer, get_pipe().tokenizer_2],
                    text_encoder=[get_pipe().text_encoder, get_pipe().text_encoder_2],
                    returned_embeddings_type=embeddings_type,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False,
                )
                positive_embeds, positive_pooled_embeds = compel([positive])
                if negative is not None and len(negative) > 0: negative_embeds, negative_pooled_embeds = compel([negative])
                positive = negative = None

            # set pipe
            kwargs                                          = {}
            kwargs["generator"]                             = generator
            kwargs["num_inference_steps"]                   = steps
            kwargs["callback_on_step_end"]                  = set_step_progress
            kwargs["callback_on_step_end_tensor_inputs"]    = ["latents"]
            match args.type:
                case "ad":
                    is_image = False
                    if ip_image is not None:
                        kwargs["ip_adapter_image"] = ip_image
                    if args.control_net is not None:
                        kwargs["conditioning_frames"] = [control_image] * frames
                    if positive is not None:    kwargs["prompt"] = positive
                    if negative is not None:    kwargs["negative_prompt"] = negative
                    kwargs["num_frames"] = frames
                    kwargs["guidance_scale"] = cfg
                    kwargs["output_type"] = "pil"
                    if height is not None: kwargs["height"] = height
                    if width is not None: kwargs["width"] = width
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
                    kwargs["num_frames"] = frames
                    kwargs["decode_chunk_size"] = decode_chunk_size
                    kwargs["motion_bucket_id"] = motion_bucket_id
                    kwargs["noise_aug_strength"] = noise_aug_strength
                    kwargs["output_type"] = "pil"
                    if height is not None: kwargs["height"] = height
                    if width is not None: kwargs["width"] = width
                case _:
                    is_image = True
                    if not args.compel:
                        if positive_embeds is not None:
                            positive_pooled_embeds = positive_embeds[0][1]["pooled_output"]
                            positive_embeds = positive_embeds[0][0]
                        if negative_embeds is not None:
                            negative_pooled_embeds = negative_embeds[0][1]["pooled_output"]
                            negative_embeds = negative_embeds[0][0]

                    if height is not None:                  kwargs["height"]                    = height
                    if width is not None:                   kwargs["width"]                     = width
                    if positive is not None:                kwargs["prompt"]                    = positive
                    if negative is not None:                kwargs["negative_prompt"]           = negative
                    if positive_embeds is not None:         kwargs["prompt_embeds"]             = positive_embeds
                    if positive_pooled_embeds is not None:  kwargs["pooled_prompt_embeds"]      = positive_pooled_embeds
                    if negative_embeds is not None:         kwargs["negative_embeds"]           = negative_embeds
                    if negative_pooled_embeds is not None:  kwargs["negative_pooled_embeds"]    = negative_pooled_embeds
                    if sigmas is not None:                  kwargs["sigmas"]                    = sigmas
                    if timesteps is not None:               kwargs["timesteps"]                 = timesteps
                    if denoising_end is not None:           kwargs["denoising_end"]             = denoising_end

                    if args.ip_adapter is not None and ip_image is not None:
                        kwargs["ip_adapter_image"] = ip_image
                    if args.control_net is not None and control_image is not None:
                        for k, v in json.loads(args.control_net).items():
                            kwargs["image"] = control_image
                            kwargs["controlnet_conditioning_scale"] = v
                    if args.type in ["sd1", "sd2", "sd3", "sdxl"]:
                        kwargs["clip_skip"] = clip_skip
                    kwargs["guidance_scale"] = cfg
                    kwargs["output_type"] = "latent"

            # inference
            output = get_pipe()(**kwargs)

            if args.compel:
                # https://github.com/damian0815/compel/issues/24
                positive_embeds = positive_pooled_embeds = negative_embeds = negative_pooled_embeds = None

            # clean up
            clean()

            # output
            if dist.get_rank() == 0:
                set_progress(100)
                if output is not None:
                    if is_image:
                        if args.type in ["sdup"]:
                            output = output.images[0]
                        else:
                            output_images = output.images
                            if args.type in ["flux"]:
                                output_images = get_pipe()._unpack_latents(output_images, height, width, get_pipe().vae_scale_factor)
                            images = convert_latent_to_image(copy.copy(output_images))
                            latents = convert_latent_to_output_latent(copy.copy(output_images))
                            return "OK", { "image": pickle_and_encode_b64(images[0]), "latent": pickle_and_encode_b64(latents) }, is_image
                    else:
                        output = output.frames[0]
                    return "OK", pickle_and_encode_b64(output), is_image
                else:
                    return "No image from pipeline", None, False


def generate_image():
    args = get_args()
    data = request.json
    dummy               = 0
    height              = data.get("height")
    width               = data.get("width")
    positive            = data.get("positive")
    negative            = data.get("negative")
    positive_embeds     = data.get("positive_embeds")
    negative_embeds     = data.get("negative_embeds")
    image               = data.get("image")
    ip_image            = data.get("ip_image")
    control_image       = data.get("control_image")
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
    sigmas              = data.get("sigmas")
    timesteps           = data.get("timesteps")
    denoising_start     = data.get("denoising_start")
    denoising_end       = data.get("denoising_end")

    print_params(data)

    if height is None:                                                  height = args.height
    if width is None:                                                   width = args.width
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
    if sigmas is not None:                                              sigmas = decode_b64_and_unpickle(sigmas)
    if timesteps is not None:                                           timesteps = decode_b64_and_unpickle(timesteps)

    params = [
        dummy,
        height,
        width,
        positive,
        negative,
        positive_embeds,
        negative_embeds,
        image,
        ip_image,
        control_image,
        latent,
        steps,
        cfg,
        controlnet_scale,
        seed,
        frames,
        decode_chunk_size,
        clip_skip,
        motion_bucket_id,
        noise_aug_strength,
        sigmas,
        timesteps,
        denoising_start,
        denoising_end,
    ]

    dist.broadcast_object_list(params, src=0)
    message, outputs, is_image = generate_image_parallel(*params)
    if is_image and args.type not in ["ad", "sdup", "svd"]:
        response = { "message": message, "output": outputs.get("image"), "latent": outputs.get("latent"), "is_image": is_image }
    else:
        response = { "message": message, "output": outputs, "is_image": is_image }
    return jsonify(response)


def run_host():
    args = get_args()
    if dist.get_rank() == 0:
        get_logger().info("Starting Flask host on rank 0")
        app.run(host="localhost", port=args.port)
    else:
        while True:
            params = [None] * 24 # len(params) of generate_image_parallel()
            get_logger().info(f"waiting for tasks")
            dist.broadcast_object_list(params, src=0)
            if params[0] is None:
                get_logger().info("Received exit signal, shutting down")
                break
            get_logger().info(f"Received task")
            generate_image_parallel(*params)


if __name__ == "__main__":
    initialize()
    run_host()
