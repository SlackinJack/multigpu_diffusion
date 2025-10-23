import copy
import gc
import json
import logging
import numpy
import os
import safetensors
import torch
import torch._dynamo
import torch.nn.functional as F
import torchvision.transforms.functional as F2
from diffusers import EulerAncestralDiscreteScheduler
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_lora_to_diffusers
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor
from optimum.quanto import freeze, qfloat8, qint2, qint4, qint8, quantize
from PIL import Image


from modules.scheduler_config import get_scheduler, get_scheduler_name,  get_scheduler_supports_setting_timesteps, get_scheduler_supports_setting_sigmas


GENERIC_HOST_ARGS = {
    "height":                       int,
    "width":                        int,
    "warm_up_steps":                int,
    "port":                         int,
    "type":                         str,
    "variant":                      str,
    "quantize_unet_to":             str,
    "quantize_encoder_to":          str,
    "quantize_misc_to":             str,
    "compile_backend":              str,    # [inductor, eager, ...]
    "compile_mode":                 str,    # [default, reduce-overhead, max-autotune, max-auto-no-cudagraphs, ...]
    "compile_options":              str,    # dict > { "triton.cudagraphs": true, ... }
    "checkpoint":                   str,    # path
    "gguf_model":                   str,    # path
    "motion_module":                str,    # path
    "motion_adapter":               str,    # path
    "motion_adapter_lora":          str,    # path
    "control_net":                  str,    # path
    "vae":                          str,    # path
    "scheduler":                    str,    # dict > { "key": value, ... }
    "lora":                         str,    # dict > { "path": scale, ... }
    "ip_adapter":                   str,    # dict > { "path": scale, ... }
    "torch_cache_limit":            int,
    "torch_accumlated_cache_limit": int,
}


GENERIC_HOST_ARGS_TOGGLES = [
    "compel",
    "enable_vae_tiling",
    "enable_vae_slicing",
    "xformers_efficient",
    "enable_model_cpu_offload",
    "enable_sequential_cpu_offload",
    "compile_unet",
    "compile_vae",
    "compile_encoder",
    "compile_fullgraph_off",
    "torch_capture_scalar",
]


def clean():
    torch.cuda.memory.empty_cache()
    gc.collect()
    return


def setup_torch_dynamo(cache_size_limit, accumulated_cache_size_limit, capture_scalar_outputs):
    if cache_size_limit == None:                cache_size_limit = 8
    if accumulated_cache_size_limit == None:    accumulated_cache_size_limit = 64
    if capture_scalar_outputs == None:          capture_scalar_outputs = False
    # TODO: config.json
    #torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = cache_size_limit
    torch._dynamo.config.accumulated_cache_size_limit = accumulated_cache_size_limit
    torch._dynamo.config.capture_scalar_outputs = capture_scalar_outputs
    return


def setup_torch_backends():
    # TODO: config.json
    # TODO: this breaks flux
    #torch.backends.cuda.enable_mem_efficient_sdp(False)
    #torch.backends.cuda.enable_flash_sdp(False)
    return


def get_logger(local_rank):
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(format=f"[Rank {local_rank}] %(message)s")
    logger = logging.getLogger(":")
    return logger


def get_torch_type(t):
    match t:
        case "fp16":    return torch.float16
        case "bf16":    return torch.bfloat16
        case _:         return torch.float32


def get_encoder_type(t):
    match t:
        case "float8":  return qfloat8
        case "int2":    return qint2
        case "int4":    return qint4
        case "int8":    return qint8
        case _:         return None


def set_scheduler(args, pipe, is_distrifuser=False):
    if is_distrifuser:  p = pipe.pipeline
    else:               p = pipe

    if args.scheduler is not None:
        args.scheduler = json.loads(args.scheduler)
        p.scheduler = get_scheduler(args.scheduler, p.scheduler.config)
    return


def load_lora(lora_dict, pipe, rank, logger):
    loras = json.loads(lora_dict)
    names = []

    for k, v in loras.items():
        logger.info(f"loading lora: {k}")
        if k.endswith(".safetensors"):  weights = safetensors.torch.load_file(k, device=f'cuda:{rank}')
        else:                           weights = torch.load(k, map_location=torch.device(f'cuda:{rank}'))
        w = k.split("/")[-1]
        a = w if not "." in w else w.split(".")[0]
        names.append(a)
        pipe.load_lora_weights(weights, weight_name=w, adapter_name=a, local_files_only=True, low_cpu_mem_usage=True)
        logger.info(f"Added LoRA (scale={v}): {k}")

    pipe.unet.set_adapters(names, list(loras.values()))
    loaded_adapters = pipe.unet.active_adapters()
    logger.info(f'Total loaded LoRAs: {len(loaded_adapters)}')
    logger.info(f'Adapters: {str(loaded_adapters)}')
    return names


def load_ip_adapter(pipe, ip_adapter_dict):
    for m, s in ip_adapter_dict.items():
        split = m.split("/")
        ip_adapter_file = split[-1]
        ip_adapter_subfolder = split[-2]
        ip_adapter_folder = m.replace(f'/{ip_adapter_subfolder}/{ip_adapter_file}', "")
        pipe.load_ip_adapter(
            ip_adapter_folder,
            subfolder=ip_adapter_subfolder,
            weight_name=ip_adapter_file,
            use_safetensors=False, # NOTE: safetensors off
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        pipe.set_ip_adapter_scale(s)
    return


def get_warmup_image():
    # image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
    image = load_image(f"{os.path.dirname(__file__)}/../resources/rocket.png") # 1024x576 pixels
    image = image.resize((768, 432), Image.Resampling.LANCZOS)
    return image


def __get_compiler_config(compiler_config):
    return config_compiler["backend"], config_compiler["mode"], config_compiler["fullgraph"]


def quantize_helper(target, pipe, quantize_to, logger, manual_module=None, is_distrifuser=False):
    quant = get_encoder_type(quantize_to)
    if quant is None:
        logger.info(f"unknown quantize type {quantize_to} - not quantizing")
        return
    logger.info(f"quantizing {target} to type {quantize_to}")
    match target:
        case "transformer":
            quantize(pipe.transformer, weights=quant)
            freeze(pipe.transformer, weights=quant)
        case "unet":
            if is_distrifuser:
                quantize(pipe.unet.model, weights=quant)
                freeze(pipe.unet.model)
            else:
                quantize(pipe.unet, weights=quant)
                freeze(pipe.unet)
        case "encoder":
            if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                quantize(pipe.text_encoder, weights=quant)
                freeze(pipe.text_encoder)
            if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
                quantize(pipe.text_encoder_2, weights=quant)
                freeze(pipe.text_encoder_2)
            if hasattr(pipe, "text_encoder_3") and pipe.text_encoder_3 is not None:
                quantize(pipe.text_encoder_3, weights=quant)
                freeze(pipe.text_encoder_3)
            if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
                quantize(pipe.image_encoder, weights=quant)
                freeze(pipe.image_encoder)
        case "manual":
            if manual_module is not None and len(manual_module) > 0:
                model = getattr(pipe, manual_module, None)
                if model is not None:
                    quantize(model, weights=quant)
                    freeze(model)
                else:
                    logger.info(f"{manual_module} not found in given pipeline - not quantizing")
                    return
            else:
                logger.info("no module given for manual target - not quantizing")
                return
        case _:
            logger.info("unknown quantize target - not quantizing")
            return
    logger.info(f"completed {quantize_to} quantization for {target}")
    return


def compile_helper(target, pipe, compile_config, logger, adapter_names=None, is_distrifuser=False):
    logger.info(f"compiling {target}")
    match target:
        case "transformer":
            if adapter_names is not None:
                pipe.transformer.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
                pipe.unload_lora_weights()
            pipe.transformer = torch.compile(pipe.transformer, **compile_config)
        case "unet":
            if is_distrifuser:
                pipe.unet.model.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
            else:
                pipe.unet.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
            pipe.unload_lora_weights()
            if is_distrifuser:
                pipe.unet.model = torch.compile(pipe.unet.model, **compile_config)
            else:
                pipe.unet = torch.compile(pipe.unet, **compile_config)
        case "vae":
            pipe.vae = torch.compile(pipe.vae, **compile_config)
        case "encoder":
            if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                pipe.text_encoder = torch.compile(pipe.text_encoder, **compile_config)
            if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
                pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, **compile_config)
            if hasattr(pipe, "text_encoder_3") and pipe.text_encoder_3 is not None:
                pipe.text_encoder_3 = torch.compile(pipe.text_encoder_3, **compile_config)
            if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
                pipe.image_encoder = torch.compile(pipe.image_encoder, **compile_config)
        case _:
            logger.info("unknown compile target - not compiling")
            return

    logger.info(f"compiled compile for {target}")
    return


# for xdit_usp only
def compile_model(pipe, logger):
    backend, mode, fullgraph = __get_compiler_config()
    logger.info(f"compiling model with {backend}:{mode}, fullgraph={fullgraph}")
    if len(mode) > 0:   pipe.model = torch.compile(pipe.model, backend=backend, mode=mode, fullgraph=fullgraph, dynamic=False)
    else:               pipe.model = torch.compile(pipe.model, backend=backend, fullgraph=fullgraph, dynamic=False)
    logger.info(f"compiled model")
    return


def print_params(params, logger):
    formatted = "Received parameters:"
    for k, v in params.items():
        formatted += f'\n{k}:{str(v)}'
    logger.info(formatted)
    return


def normalize_latent(x, max_val=3.0):
    max_val = torch.tensor(max_val)
    x = x.detach().clone()
    for i in range(x.shape[0]):
        x[[i], :] = torch.sqrt(max_val**2 - 1) * x[[i], :] / torch.sqrt(torch.add(x[[i], :]**2, max_val**2 - 1))
        for chl in range(4):
            if x[i, chl, :, :].std() > 1.0:
                x[i, chl, :, :] /= x[i, chl, :, :].std()
    return x


def process_input_latent(latents, pipe, dtype, device, timestep=None, is_distrifuser=False):
    if is_distrifuser:  p = pipe.pipeline
    else:               p = pipe

    latents = add_alpha_to_latent(latents)
    latents = latents.to(pipe.device)

    if timestep == None:    latents = p.scheduler.scale_model_input(latents, timestep=p.scheduler.timesteps[-1])
    else:                   latents = p.scheduler.scale_model_input(latents, timestep=timestep)
    latents = latents * p.vae.config.scaling_factor

    target = 255
    latents = normalize_latent(latents, max_val=target)
    #latents = latents * (68.5 / 100)

    latents = latents.to(dtype)
    return latents


def add_alpha_to_latent(latents):
    if latents.shape[1] == 3:
        alpha = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], device=latents.device)
        latents = torch.cat((latents, alpha), dim=1)
    return latents


def remove_alpha_from_latent(latents):
    if latents.shape[1] == 4:
        latents = latents[:, :3, :, :]
    return latents


def process_latent_for_output(latents, pipe, is_latent_output):
    default_vae_dtype = copy.copy(pipe.vae.dtype)
    pipe.vae = pipe.vae.to(torch.float32)
    latents = latents.to(torch.float32)
    latents = latents / pipe.vae.config.scaling_factor
    latents = add_alpha_to_latent(latents)
    latents = latents * (85.0 / 100)
    if is_latent_output:
        pipe.vae = pipe.vae.to(default_vae_dtype)
        return latents
    latents = pipe.vae.decode(latents, return_dict=False)[0]
    latents = pipe.image_processor.postprocess(latents, output_type="pil")
    pipe.vae = pipe.vae.to(default_vae_dtype)
    return latents


def convert_latent_to_output_latent(latents, pipe, is_distrifuser=False):
    if is_distrifuser:  p = pipe.pipeline
    else:               p = pipe
    latents = process_latent_for_output(latents, p, True)
    return latents


def convert_latent_to_image(latents, pipe, is_distrifuser=False):
    if is_distrifuser:  p = pipe.pipeline
    else:               p = pipe
    latents = process_latent_for_output(latents, p, False)
    return latents
