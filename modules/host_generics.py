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
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor
from optimum.quanto import freeze, qfloat8, qint2, qint4, qint8, quantize
from PIL import Image


config          = json.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../config.json"))
config_compiler = config["compiler"]


GENERIC_HOST_ARGS = {
    "height":               int,
    "width":                int,
    "warm_up_steps":        int,
    "port":                 int,
    "type":                 str,
    "variant":              str,
    "quantize_to":          str,
    "checkpoint":           str,    # path
    "gguf_model":           str,    # path
    "motion_module":        str,    # path
    "motion_adapter":       str,    # path
    "motion_adapter_lora":  str,    # path
    "control_net":          str,    # path
    "vae":                  str,    # path
    "scheduler":            str,    # json dict > { "key": value, ... }
    "lora":                 str,    # json dict > { "path": scale, ... }
    "ip_adapter":           str,    # json dict > { "path": scale, ... }
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
    "compile_text_encoder",
]


def clean():
    torch.cuda.memory.empty_cache()
    gc.collect()
    return


def setup_torch_dynamo():
    # TODO: config.json
    #torch._dynamo.config.suppress_errors = True
    #torch._dynamo.config.capture_scalar_outputs = False
    torch._dynamo.config.cache_size_limit = int(8 * 4)
    torch._dynamo.config.accumulated_cache_size_limit = int(64 * 4)
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


def do_quantization(model, desc, quantize_to, logger):
    logger.info(f"quantizing {desc} to {quantize_to}")
    quant = get_encoder_type(quantize_to)
    quantize(model, weights=quant)
    freeze(model)
    logger.info(f"completed {quantize_to} quantization for {desc}")
    return


def load_lora(lora_dict, pipe, rank, logger, is_quantized):
    loras = json.loads(lora_dict)
    names = []

    if is_quantized:
        logger.info("It looks like you are using LoRAs with quantization. This may degrade image quality.")

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


def __get_compiler_config():
    global config_compiler
    return config_compiler["backend"], config_compiler["mode"], config_compiler["fullgraph"]


def compile_unet(pipe, adapter_names, logger, is_distrifuser=False):
    backend, mode, fullgraph = __get_compiler_config()
    kwargs = {"backend": backend, "fullgraph": fullgraph, "dynamic": False}
    if len(mode) > 0: kwargs["mode"] = mode
    logger.info(f"compiling unet with {backend}:{mode}, fullgraph={fullgraph}")
    if adapter_names:
        if is_distrifuser:  pipe.unet.model.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
        else:               pipe.unet.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
        pipe.unload_lora_weights()
    if is_distrifuser:      pipe.unet.model = torch.compile(pipe.unet.model, **kwargs)
    else:                   pipe.unet = torch.compile(pipe.unet, **kwargs)
    logger.info(f"compiled unet")
    return


def compile_transformer(pipe, adapter_names, logger):
    backend, mode, fullgraph = __get_compiler_config()
    logger.info(f"compiling transformer with {backend}:{mode}, fullgraph={fullgraph}")
    if adapter_names:
        pipe.transformer.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
    pipe.unload_lora_weights()
    if len(mode) > 0:   pipe.transformer = torch.compile(pipe.transformer, backend=backend, mode=mode, fullgraph=fullgraph, dynamic=False)
    else:               pipe.transformer = torch.compile(pipe.transformer, backend=backend, fullgraph=fullgraph, dynamic=False)
    logger.info(f"compiled transformer")
    return


# for xdit_usp only
def compile_model(pipe, logger):
    backend, mode, fullgraph = __get_compiler_config()
    logger.info(f"compiling model with {backend}:{mode}, fullgraph={fullgraph}")
    if len(mode) > 0:   pipe.model = torch.compile(pipe.model, backend=backend, mode=mode, fullgraph=fullgraph, dynamic=False)
    else:               pipe.model = torch.compile(pipe.model, backend=backend, fullgraph=fullgraph, dynamic=False)
    logger.info(f"compiled model")
    return


def compile_vae(pipe, logger):
    backend, mode, fullgraph = __get_compiler_config()
    logger.info(f"compiling vae with {backend}:{mode}, fullgraph={fullgraph}")
    if len(mode) > 0:   pipe.vae = torch.compile(pipe.vae, backend=backend, mode=mode, fullgraph=fullgraph, dynamic=False)
    else:               pipe.vae = torch.compile(pipe.vae, backend=backend, fullgraph=fullgraph, dynamic=False)
    logger.info(f"compiled vae")
    return


def compile_text_encoder(pipe, logger):
    backend, mode, fullgraph = __get_compiler_config()
    logger.info(f"compiling text encoder with {backend}:{mode}, fullgraph={fullgraph}")
    if len(mode) > 0:   pipe.text_encoder = torch.compile(pipe.text_encoder, backend=backend, mode=mode, fullgraph=fullgraph, dynamic=False)
    else:               pipe.text_encoder = torch.compile(pipe.text_encoder, backend=backend, fullgraph=fullgraph, dynamic=False)
    logger.info(f"compiled text encoder")
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


def process_input_latent(latents, scheduler_name, pipe, dtype, device, is_distrifuser=False):
    if is_distrifuser:  p = pipe.pipeline
    else:               p = pipe

    latents = add_alpha_to_latent(latents)

    latents = p.scheduler.scale_model_input(latents, timestep=p.scheduler.timesteps[-1])
    latents = latents * p.vae.config.scaling_factor

    target = 8
    latents = normalize_latent(latents, max_val=target)

    latents = latents * (94 / 192)

    latents = latents.to(device)
    latents = latents.to(dtype)
    return latents


def deduplicate_timesteps(timesteps, logger):
    clamped = False
    max_timestep = timesteps[0]
    timesteps = timesteps[::-1]
    out = [timesteps[0]]
    for i in timesteps[1:]:
        last = out[-1]
        while i <= last:
            i += 1
        if not i > max_timestep:
            out.append(i)
        else:
            clamped = True
            break
    out = out[::-1]
    if clamped:
        logger.info(f"timesteps were clamped to {len(out)} steps")
    return out


def set_timesteps(pipe, latents, scheduler_name, steps, denoise, sigmas, timesteps, logger, is_distrifuser=False):
    if is_distrifuser:  p = pipe.pipeline
    else:               p = pipe

    # preference order: timesteps > sigmas > betas

    use_mu = [] # deis, dpmpp_2m, dpmpp_sde, unipc, needs diffusers >= 0.35.1
    use_timesteps = ["ddpm", "dpmpp_2m", "dpmpp_sde", "euler", "heun", "tcd"]
    use_sigmas = ["euler"]
    use_betas = ["ddim", "deis", "dpm_sde", "euler_a", "dpm_2", "dpm_2_a", "lms", "pndm", "unipc"]
    bypass_beta_rescale = ["euler_a"]
    no_denoise = ["ipndm"]

    if scheduler_name in no_denoise:
        logger.info(f"scheduler {scheduler_name} does not support any types of denoising - ignoring denoise")
        return None, None
    elif scheduler_name in use_betas:
        # TODO: dial this in
        denoise_rescale = 1 # 4 / 5
        if not scheduler_name in bypass_beta_rescale:
            denoise = denoise * denoise_rescale
        if latents is not None:
            latents_rescale = (50 / 32) * (45 * denoise_rescale / 32) - (7 * denoise_rescale / 32)
            latents = latents / denoise_rescale * latents_rescale
        logger.info(f"using scheduler betas for denoise (this may result in blurry images)")
        # reset scheduler first
        current_config = {}
        for k, v in p.scheduler.config.items():
            if k not in ["beta_start", "beta_end"]:
                current_config[k] = v
        p.scheduler = p.scheduler.from_config(current_config)
        # set new scheduler
        new_config = {}
        for k, v in p.scheduler.config.items():
            new_config[k] = v
        new_config["beta_start"] = new_config["beta_start"] * denoise
        new_config["beta_end"] = new_config["beta_end"] * denoise
        p.scheduler = p.scheduler.from_config(new_config)
        return latents, { "num_inference_steps": int(steps) }
    elif denoise == 1.00:
        return None, None
    elif scheduler_name in use_mu:
        p.scheduler.set_timesteps(mu=denoise)
        return latents, None
    else:
        is_timestep = True
        if timesteps is not None:
            if scheduler_name not in use_timesteps:
                logger.info(f"scheduler {scheduler_name} does not support timesteps - ignoring denoise")
                return None, None
            out = [int(t * denoise) for t in timesteps]
        elif sigmas is not None:
            is_timestep = False
            if scheduler_name not in use_sigmas:
                logger.info(f"scheduler {scheduler_name} does not support sigmas - ignoring denoise")
                return None, None
            out = [float(s * denoise) for s in sigmas]
        else:
            if scheduler_name in use_timesteps:
                t = p.scheduler.timesteps
                t_max = int(t[0])
                t_min = int(t[-1])
            elif scheduler_name in use_sigmas:
                is_timestep = False
                t = p.scheduler.sigmas
                t_max = float(t[0])
                t_min = float(t[-1])
            else:
                return None, None
            out = []
            for i in range(steps):
                next_val = i * (t_min + t_max) * denoise / steps
                if is_timestep: next_val = int(next_val)
                else:           next_val = float(next_val)
                if len(out) > 0 and out[i - 1] >= next_val: out.append(t_min if next_val <= t_min else next_val)
                else:               out.insert(0, t_min if next_val <= t_min else next_val)

        latents_rescale = (61 / 32) * (35 / 32 * (1 - denoise))
        latents = latents * latents_rescale
        if is_timestep: out = deduplicate_timesteps(out, logger)
        if is_timestep: return latents, { "timesteps": out }
        else:           return latents, { "sigmas": out }


def add_alpha_to_latent(latents):
    if latents.shape[1] == 3:
        alpha = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], device=latents.device)
        latents = torch.cat((latents, alpha), dim=1)
    return latents


def remove_alpha_from_latent(latents):
    if latents.shape[1] == 4:
        latents = latents[:, :3, :, :]
    return latents


def convert_latent_to_output_latent(latents, pipe, is_distrifuser=False):
    if is_distrifuser:  p = pipe.pipeline
    else:               p = pipe
    latents = latents.to(torch.float32)
    latents = latents / p.vae.config.scaling_factor
    latents = add_alpha_to_latent(latents)
    return latents


def convert_latent_to_image(latents, pipe, is_distrifuser=False):
    if is_distrifuser:  p = pipe.pipeline
    else:               p = pipe
    default_pipe_dtype = copy.deepcopy(p.vae.dtype)
    latents = latents.to(torch.float32)
    latents = latents / p.vae.config.scaling_factor
    latents = add_alpha_to_latent(latents)
    latents = p.vae.decode(latents, return_dict=False)[0]
    image = p.image_processor.postprocess(latents, output_type="pil")
    p.vae = p.vae.to(default_pipe_dtype)
    return image
