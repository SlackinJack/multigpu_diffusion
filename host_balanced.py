import argparse
import base64
import io
import json
import os
import sys
import torch
import traceback
from diffusers import (
    AnimateDiffControlNetPipeline,
    AnimateDiffPipeline,
    FluxControlNetPipeline,
    FluxPipeline,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
    WanPipeline,
    WanImageToVideoPipeline,
    ZImagePipeline,
    # ZImageControlNetPipeline,
)
from diffusers.utils import load_image
from flask import Flask, request, jsonify
from PIL import Image


from modules.host_generics import *
from modules.scheduler_config import *
from modules.utils import *


app = Flask(__name__)


def __initialize_distributed_environment():
    set_local_rank(0)
    set_logger()
    set_initialized(True)
    return


args = None
def __run_host():
    global args
    parser = argparse.ArgumentParser()
    # generic
    for k, v in GENERIC_HOST_ARGS.items():  parser.add_argument(f"--{k}", type=v, default=None)
    for e in GENERIC_HOST_ARGS_TOGGLES:     parser.add_argument(f"--{e}", action="store_true")
    args = parser.parse_args()

    log("Starting Flask host")
    app.run(host="localhost", port=args.port)
    return


@app.route("/<path>", methods=["GET", "POST"])
def handle_path(path):
    match path:
        # status
        case "initialize":
            return get_initialized_flask()
        case "progress":
            return get_progress_flask()

        # generation
        case "apply":
            return __apply_pipeline_parallel(request.json)
        case "generate":
            return __generate_image_parallel(request.json)
        case "offload":
            return __offload_modules_parallel()
        case "close":
            sys.exit()

        case _:
            return "", 404


def __offload_modules_parallel():
    return "Operation not supported", 500


applied = None
def __apply_pipeline_parallel(data):
    global applied

    try:
        # models
        pipeline_type                   = data.get("pipeline_type")
        variant                         = data.get("variant")
        checkpoint                      = data.get("checkpoint")
        unet                            = data.get("unet")
        unet_config                     = data.get("unet_config")
        transformer                     = data.get("transformer")
        transformer_config              = data.get("transformer_config")
        vae                             = data.get("vae")
        vae_fp16                        = data.get("vae_fp16")
        vae_config                      = data.get("vae_config")
        control_net                     = data.get("control_net")
        control_net_config              = data.get("control_net_config")
        motion_adapter                  = data.get("motion_adapter")
        motion_module                   = data.get("motion_module")
        motion_config                   = data.get("motion_config")
        ip_adapter                      = data.get("ip_adapter")
        lora                            = data.get("lora")

        # compile
        compile_config                  = data.get("compile_config")
        torch_config                    = data.get("torch_config")

        # quantization
        quantization_config             = data.get("quantization_config")

        # memory
        enable_vae_slicing              = data.get("enable_vae_slicing")
        enable_vae_tiling               = data.get("enable_vae_tiling")
        xformers_efficient              = data.get("xformers_efficient")
        enable_sequential_cpu_offload   = data.get("enable_sequential_cpu_offload")
        enable_model_cpu_offload        = data.get("enable_model_cpu_offload")

        print_params(data)

        # checks
        assert not (pipeline_type == "ad" and motion_adapter is None and motion_module is None), "AnimateDiff requires providing a motion adapter/module."

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                if str(data) == str(applied): return "", 200
                set_pipe(None)

                PipelineClass = None

                # dynamo tweaks
                if torch_config is not None:
                    torch_cache_limit = torch_config.get("torch_cache_limit")
                    torch_accumlated_cache_limit = torch_config.get("torch_accumlated_cache_limit")
                    torch_capture_scalar = torch_config.get("torch_capture_scalar")
                    setup_torch_dynamo(torch_cache_limit, torch_accumlated_cache_limit, torch_capture_scalar)

                # torch tweaks
                setup_torch_backends()

                # update globals
                set_vae_fp16(vae_fp16 is not None)
                set_torch_dtype(get_torch_type(variant))

                # set pipeline
                log(f"Initializing pipeline")
                kwargs = {}
                kwargs["torch_dtype"] = get_torch_dtype()
                kwargs["use_safetensors"] = True
                kwargs["local_files_only"] = True
                kwargs["low_cpu_mem_usage"] = True
                kwargs["add_watermarker"] = False
                kwargs["device_map"] = "balanced"

                # quantize
                is_quantized = False
                mappings = {}
                if quantization_config is not None:
                    quantize_unet                       = quantization_config.get("quantize_unet")
                    quantize_encoder                    = quantization_config.get("quantize_encoder")
                    quantize_vae                        = quantization_config.get("quantize_vae")
                    quantize_tokenizer                  = quantization_config.get("quantize_tokenizer")
                    quantize_misc                       = quantization_config.get("quantize_misc")
                    if quantize_unet is not None:       mappings.update(get_quant_mapping("unet", quantize_unet))
                    if quantize_encoder is not None:    mappings.update(get_quant_mapping("encoder", quantize_encoder))
                    if quantize_vae is not None:        mappings.update(get_quant_mapping("vae", quantize_vae))
                    if quantize_tokenizer is not None:  mappings.update(get_quant_mapping("tokenizer", quantize_tokenizer))
                    if quantize_misc is not None:       mappings.update(get_quant_mapping("misc", quantize_misc))
                    if len(list(mappings.keys())) > 0:
                        is_quantized = True
                        kwargs["quantization_config"] = get_pipe_quant_config(mappings)

                # set control net
                controlnet_model = None
                if control_net is not None and pipeline_type not in ["sdup", "svd"]:
                    kwargs["controlnet"] = load_model(control_net, control_net_config, "ControlNetModel")

                # set unet
                if unet is not None:
                    kwargs["unet"] = load_model(unet, unet_config, "UNet2DConditionModel")

                # set transformer
                if transformer is not None:
                    match pipeline_type:
                        case "flux":
                            kwargs["transformer"] = load_model(transformer, transformer_config, "FluxTransformer2DModel")
                        case "sd3":
                            kwargs["transformer"] = load_model(transformer, transformer_config, "SD3Transformer2DModel")
                        case "zimage":
                            kwargs["transformer"] = load_model(transformer, transformer_config, "ZImageTransformer2DModel")

                # set vae
                if vae is not None and pipeline_type not in ["ad", "svd"]:
                    kwargs["vae"] = load_model(vae, vae_config, "AutoencoderKL")

                # set motion_adapter
                if (motion_module is not None or motion_adapter is not None) and pipeline_type in ["ad"]:
                    if motion_module is not None:
                        kwargs["motion_adapter"] = load_model(motion_module, motion_config, "MotionAdapter")
                    else:
                        kwargs["motion_adapter"] = load_model(motion_adapter, motion_config, "MotionAdapter")

                match pipeline_type:
                    case "ad":
                        PipelineClass = AnimateDiffControlNetPipeline if control_net is not None else AnimateDiffPipeline
                    case "flux":
                        PipelineClass = FluxControlNetPipeline if control_net is not None else FluxPipeline
                    case "sd1":
                        PipelineClass = StableDiffusionControlNetPipeline if control_net is not None else StableDiffusionPipeline
                    case "sd2":
                        PipelineClass = StableDiffusionControlNetPipeline if control_net is not None else StableDiffusionPipeline
                    case "sd3":
                        PipelineClass = StableDiffusion3ControlNetPipeline if control_net is not None else StableDiffusion3Pipeline
                    case "sdup":
                        PipelineClass = StableDiffusionUpscalePipeline
                    case "sdxl":
                        PipelineClass = StableDiffusionXLControlNetPipeline if control_net is not None else StableDiffusionXLPipeline
                    case "svd":
                        PipelineClass = StableVideoDiffusionPipeline
                    case "want2v":
                        PipelineClass = WanPipeline
                    case "wani2v":
                        PipelineClass = WanImageToVideoPipeline
                    case "zimage":
                        # PipelineClass = ZImageControlNetPipeline if control_net is not None else ZImagePipeline
                        PipelineClass = ZImagePipeline
                    case _: raise NotImplementedError

                # init pipe
                set_pipe(PipelineClass.from_pretrained(checkpoint, **kwargs))
                del kwargs
                log("Pipeline initialized")

                # set ipadapter
                if ip_adapter is not None:
                    load_ip_adapter(ip_adapter)

                # set memory saving
                if pipeline_type not in ["svd"]:
                    if enable_vae_slicing:         get_pipe().vae.enable_slicing()
                    if enable_vae_tiling:          get_pipe().vae.enable_tiling()
                    if pipeline_type not in ["flux"]:
                        if xformers_efficient:     get_pipe().enable_xformers_memory_efficient_attention()
                if enable_sequential_cpu_offload:  log("sequential CPU offload not supported - ignoring")
                if enable_model_cpu_offload:       log("model CPU offload not supported - ignoring")

                # set lora
                adapter_names = None
                if lora is not None and pipeline_type not in ["ad", "sdup", "svd"]:
                    adapter_names = load_lora(lora)

                # compiles
                if compile_config is not None:
                    compile_unet = compile_config.get("compile_unet")
                    compile_vae = compile_config.get("compile_vae")
                    compile_encoder = compile_config.get("compile_encoder")
                    compile_backend = compile_config.get("compile_backend")
                    compile_mode = compile_config.get("compile_mode")
                    compile_options = compile_config.get("compile_options")
                    compile_fullgraph_off = compile_config.get("compile_fullgraph_off")

                    if compile_mode is not None and compile_options is not None:
                        log("Compile mode and options are both defined, will ignore compile mode.")
                        compile_mode = None
                    compiler_config                                 = {}
                    compiler_config["fullgraph"]                    = (compile_fullgraph_off is None or compile_fullgraph_off == False)
                    compiler_config["dynamic"]                      = False
                    if compile_backend is not None:            compiler_config["backend"] = compile_backend
                    if compile_mode is not None:               compiler_config["mode"] = compile_mode
                    if compile_options is not None:            compiler_config["options"] = json.loads(compile_options)

                    if compile_unet:
                        if pipeline_type in ["flux", "sd3", "wani2v", "want2v", "zimage"]:
                            compile_helper("transformer", compiler_config, adapter_names=adapter_names)
                        else:
                            compile_helper("unet", compiler_config, adapter_names=adapter_names)
                    if compile_vae:                            compile_helper("vae", compiler_config)
                    if compile_encoder:                        compile_helper("encoder", compiler_config)

                # set models to eval mode
                setup_evals()

                # clean up
                clean()

                # complete
                log("Model initialization completed")
                print_mem_usage()
                applied = data
                return "", 200
    except:
        log(traceback.format_exc())
        return "", 500


def __generate_image_parallel(data):
    global applied

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
    ip_adapter_scale    = data.get("ip_adapter_scale")
    seed                = data.get("seed")
    frames              = data.get("frames")
    decode_chunk_size   = data.get("decode_chunk_size")
    clip_skip           = data.get("clip_skip")
    motion_bucket_id    = data.get("motion_bucket_id")
    noise_aug_strength  = data.get("noise_aug_strength")
    denoising_start     = data.get("denoising_start")
    denoising_end       = data.get("denoising_end")
    scheduler           = data.get("scheduler")
    use_compel          = data.get("use_compel")

    print_params(data)

    if positive is not None and len(positive) == 0:                     positive = None
    if negative is not None and len(negative) == 0:                     negative = None
    if image is None and positive is None and positive_embeds is None:  jsonify({ "message": "No input provided", "output": None, "is_image": False })
    if positive is not None and positive_embeds is not None:            jsonify({ "message": "Provide only one positive input", "output": None, "is_image": False })
    if negative is not None and negative_embeds is not None:            jsonify({ "message": "Provide only one negative input", "output": None, "is_image": False })
    if image is not None:                                               image = image = Image.open(io.BytesIO(decode_b64_and_unpickle(image)))
    if ip_image is not None:                                            ip_image = decode_b64_and_unpickle(ip_image)
    if control_image is not None:                                       control_image = decode_b64_and_unpickle(control_image)
    if latent is not None:                                              latent = decode_b64_and_unpickle(latent)
    if positive_embeds is not None:                                     positive_embeds = decode_b64_and_unpickle(positive_embeds)
    if negative_embeds is not None:                                     negative_embeds = decode_b64_and_unpickle(negative_embeds)
    if denoising_start is None or denoising_start < 0:                  denoising_start = 0
    if denoising_end is None or denoising_end > steps:                  denoising_end = steps

    pipeline_type = applied.get("pipeline_type")

    # checks
    if pipeline_type in ["sdup", "svd", "wani2v"] and image is None:
        return { "message": "No image provided for an image pipeline.", "output": None, "is_image": False }
    if applied.get("ip_adapter") is not None and ip_image is None:
        return { "message": "No IPAdapter image provided for a IPAdapter-loaded pipeline", "output": None, "is_image": False }
    if applied.get("control_net") is not None and control_image is None:
        return { "message": "No ConstrolNet image provided for a ControlNet-loaded pipeline", "output": None, "is_image": False }
    if use_compel is not None and use_compel == True:
        return { "message": "This backend does not support Compel. You must encode the prompt(s) first using Compel externally.", "output": None, "is_image": False }

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            torch.cuda.reset_peak_memory_stats()
            set_progress(0)

            # load image
            if pipeline_type in ["sdup", "svd", "wani2v"]:
                image = load_image(image)
            if ip_image is not None and applied.get("ip_adapter") is not None:
                ip_image = load_image(ip_image)
            if control_image is not None and applied.get("control_net") is not None:
                control_image = load_image(control_image)

            # set scheduler
            if scheduler is not None:   set_scheduler(scheduler)
            if latent is not None:      latent = process_input_latent(latent)
            set_scheduler_timesteps(denoising_start)

            # progress bar
            def set_step_progress(pipe, index, timestep, callback_kwargs):
                global get_logger, get_scheduler_progressbar_offset_index, set_progress
                nonlocal steps
                the_index = get_scheduler_progressbar_offset_index(pipe.scheduler, index)
                log(str(callback_kwargs["latents"]))
                set_progress(int(the_index / steps * 100))
                return callback_kwargs

            # set seed
            generator = torch.Generator(device="cpu").manual_seed(seed)

            # conditioning
            positive_pooled_embeds = None
            negative_pooled_embeds = None
            if positive_embeds is not None:
                positive_pooled_embeds = positive_embeds[0][1]["pooled_output"]
                positive_embeds = positive_embeds[0][0]
            if negative_embeds is not None:
                negative_pooled_embeds = negative_embeds[0][1]["pooled_output"]
                negative_embeds = negative_embeds[0][0]

            # set pipe
            kwargs                                          = {}
            kwargs["generator"]                             = generator
            kwargs["num_inference_steps"]                   = steps
            kwargs["callback_on_step_end"]                  = set_step_progress
            kwargs["callback_on_step_end_tensor_inputs"]    = ["latents"]
            match pipeline_type:
                case "ad":
                    if ip_image is not None:
                        kwargs["ip_adapter_image"] = ip_image
                    if applied.get("control_net") is not None:
                        kwargs["conditioning_frames"] = [control_image] * frames
                    if positive is not None:    kwargs["prompt"] = positive
                    if negative is not None:    kwargs["negative_prompt"] = negative
                    kwargs["num_frames"] = frames
                    kwargs["guidance_scale"] = cfg
                    kwargs["output_type"] = "pil"
                    if height is not None: kwargs["height"] = height
                    if width is not None: kwargs["width"] = width
                case "sdup":
                    if positive is not None:    kwargs["prompt"] = positive
                    if negative is not None:    kwargs["negative_prompt"] = negative
                    if image is not None:       kwargs["image"] = image
                    kwargs["guidance_scale"] = cfg
                    kwargs["output_type"] = "pil"
                case "svd":
                    if image is not None: kwargs["image"] = image
                    kwargs["num_frames"] = frames
                    kwargs["decode_chunk_size"] = decode_chunk_size
                    kwargs["motion_bucket_id"] = motion_bucket_id
                    kwargs["noise_aug_strength"] = noise_aug_strength
                    kwargs["output_type"] = "pil"
                    if height is not None: kwargs["height"] = height
                    if width is not None: kwargs["width"] = width
                case "wani2v":
                    if image is not None: kwargs["image"] = image
                    kwargs["output_type"] = "pil"
                    if height is not None:                  kwargs["height"]                    = height
                    if width is not None:                   kwargs["width"]                     = width
                    if positive is not None:                kwargs["prompt"]                    = positive
                    if negative is not None:                kwargs["negative_prompt"]           = negative
                    # if positive_embeds is not None:         kwargs["prompt_embeds"]             = positive_embeds
                    # if positive_pooled_embeds is not None:  kwargs["pooled_prompt_embeds"]      = positive_pooled_embeds
                    # if negative_embeds is not None:         kwargs["negative_embeds"]           = negative_embeds
                    # if negative_pooled_embeds is not None:  kwargs["negative_pooled_embeds"]    = negative_pooled_embeds
                case _:
                    if latent is not None:
                        kwargs["latents"] = latent
                    else:
                        if height is not None:              kwargs["height"]                    = height
                        if width is not None:               kwargs["width"]                     = width
                    if positive is not None:                kwargs["prompt"]                    = positive
                    if negative is not None:                kwargs["negative_prompt"]           = negative
                    if positive_embeds is not None:         kwargs["prompt_embeds"]             = positive_embeds
                    if positive_pooled_embeds is not None:  kwargs["pooled_prompt_embeds"]      = positive_pooled_embeds
                    if negative_embeds is not None:         kwargs["negative_embeds"]           = negative_embeds
                    if negative_pooled_embeds is not None:  kwargs["negative_pooled_embeds"]    = negative_pooled_embeds
                    if denoising_end is not None:           kwargs["denoising_end"]             = float(denoising_end / steps)

                    if applied.get("ip_adapter") is not None and ip_image is not None:
                        kwargs["ip_adapter_image"] = ip_image
                        if ip_adapter_scale is not None:    get_pipe().set_ip_adapter_scale(scale)
                        else:                               get_pipe().set_ip_adapter_scale(1.0)
                    if applied.get("control_net") is not None and control_image is not None:
                        kwargs["image"] = control_image
                        if controlnet_scale is not None:    kwargs["controlnet_conditioning_scale"] = controlnet_scale
                        else:                               kwargs["controlnet_conditioning_scale"] = 1.0
                    if pipeline_type in ["sd1", "sd2", "sd3", "sdxl"]:
                        kwargs["clip_skip"] = clip_skip
                    kwargs["guidance_scale"] = cfg
                    kwargs["output_type"] = "latent"

            # inference
            output = get_pipe()(**kwargs)

            # clean up
            clean()

            # output
            set_progress(100)
            if output is not None:
                if get_is_image_model(pipeline_type):
                    if pipeline_type in ["sdup"]:
                        output = output.images[0]
                    else:
                        output_images = output.images
                        if pipeline_type in ["flux"]:
                            output_images = get_pipe()._unpack_latents(output_images, height, width, get_pipe().vae_scale_factor)
                        flag = False
                        if get_pipe().vae.device == torch.device("cpu"):
                            flag = True
                            get_pipe().vae = get_pipe().vae.to(device=output_images.device)
                        images = convert_latent_to_image(output_images)
                        latents = convert_latent_to_output_latent(output_images)
                        if flag:
                            get_pipe().vae = get_pipe().vae.to(device="cpu")
                        return { "message": "OK", "output": pickle_and_encode_b64(images[0]), "latent": pickle_and_encode_b64(latents), "is_image": True }
                else:
                    output = output.frames[0]
                return { "message": "OK", "output": pickle_and_encode_b64(output), "is_image": False }
            else:
                return { "message": "No image from pipeline", "output": None, "is_image": False }


if __name__ == "__main__":
    __initialize_distributed_environment()
    __run_host()
