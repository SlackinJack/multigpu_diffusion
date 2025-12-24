import copy
import gc
import json
import logging
import numpy
import os
import safetensors
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    GGUFQuantizationConfig,
    FluxTransformer2DModel,
    MotionAdapter,
    PipelineQuantizationConfig,
    SD3Transformer2DModel,
    StableDiffusionPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    ZImageTransformer2DModel,
)
from diffusers import BitsAndBytesConfig as BitsAndBytesConfigD
from diffusers import TorchAoConfig as TorchAoConfigD
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_lora_to_diffusers
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor
from torchao.quantization import FPXWeightOnlyConfig, UIntXWeightOnlyConfig
from transformers import BitsAndBytesConfig as BitsAndBytesConfigT
from transformers import TorchAoConfig as TorchAoConfigT
from PIL import Image


from modules.scheduler_config import get_scheduler, get_scheduler_name,  get_scheduler_supports_setting_timesteps, get_scheduler_supports_setting_sigmas


GENERIC_HOST_ARGS = {
    # models
    "unet":                         str,    # path
    "unet_config":                  str,    # path
    "transformer":                  str,    # path
    "transformer_config":           str,    # path
    "vae":                          str,    # path
    "vae_config":                   str,    # path
    "motion_module":                str,    # path
    "motion_adapter":               str,    # path
    "motion_lora":                  str,    # path
    "motion_config":                str,    # path
    "control_net":                  str,    # dict > { "path": scale }
    "control_net_config":           str,    # path
    "ip_adapter":                   str,    # dict > { "path": scale, ... }
    "lora":                         str,    # dict > { "path": scale, ... }

    # image
    "scheduler":                    str,    # dict > { "key": value, ... }

    # video

    # optimization
    "torch_cache_limit":            int,
    "torch_accumlated_cache_limit": int,
    "compile_backend":              str,    # [inductor, eager, ...]
    "compile_mode":                 str,    # [default, reduce-overhead, max-autotune, max-auto-no-cudagraphs, ...]
    "compile_options":              str,    # dict > { "triton.cudagraphs": true, ... }
    "quantize_unet":                str,
    "quantize_encoder":             str,
    "quantize_vae":                 str,
    "quantize_tokenizer":           str,
    "quantize_scheduler":           str,
    "quantize_misc":                str,

    # host
    "checkpoint":                   str,    # path
    "type":                         str,    # sd1, sd2, sd3, sdxl, etc.
    "port":                         int,
    "variant":                      str,
    "warm_up_steps":                int,
}


GENERIC_HOST_ARGS_TOGGLES = [
    # optimization
    "torch_capture_scalar",
    "compile_unet",
    "compile_vae",
    "compile_encoder",
    "compile_fullgraph_off",
    "enable_vae_tiling",
    "enable_vae_slicing",
    "xformers_efficient",
    "enable_model_cpu_offload",
    "enable_sequential_cpu_offload",

    # host
    "compel",
]


initialized = False
def set_initialized(is_initialized):
    global initialized
    initialized = is_initialized
    return


def get_initialized():
    global initialized
    return initialized


def get_initialized_flask():
    if get_initialized():   return "OK", 200
    else:                   return "WAIT", 202


progress = 0
def set_progress(progress_in):
    global progress
    progress = progress_in
    return


def get_progress():
    global progress
    return progress


def get_progress_flask():
    global progress
    return str(progress), 200


pipe = None
def set_pipe(pipe_in):
    global pipe
    pipe = pipe_in
    return


def get_pipe():
    global pipe
    return pipe


def clean():
    torch.cuda.memory.empty_cache()
    gc.collect()
    return


def setup_torch_dynamo(cache_size_limit, accumulated_cache_size_limit, capture_scalar_outputs):
    if cache_size_limit is not None:                torch._dynamo.config.cache_size_limit               = cache_size_limit
    if accumulated_cache_size_limit is not None:    torch._dynamo.config.accumulated_cache_size_limit   = accumulated_cache_size_limit
    if capture_scalar_outputs is not None:          torch._dynamo.config.capture_scalar_outputs         = capture_scalar_outputs
    #torch._dynamo.config.suppress_errors           = True
    return


def setup_torch_backends():
    # TODO: config
    # TODO: this breaks flux
    #torch.backends.cuda.enable_mem_efficient_sdp(False)
    #torch.backends.cuda.enable_flash_sdp(False)
    return


logger = None
def set_logger(local_rank):
    global logger
    if logger is None:
        logging.root.setLevel(logging.NOTSET)
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(f"[Rank {str(local_rank)}]")
    return


def get_logger():
    global logger
    return logger


def get_torch_type(t):
    match t:
        case "bf16":    return torch.bfloat16
        case "fp8":     return torch.float8_e4m3fn
        case "fp16":    return torch.float16
        case "fp32":    return torch.float32
        case "fp64":    return torch.float64
        case "cp32":    return torch.complex32
        case "cp64":    return torch.complex64
        case "cp128":   return torch.complex128
        case "int1":    return torch.uint1
        case "int2":    return torch.uint2
        case "int3":    return torch.uint3
        case "int4":    return torch.uint4
        case "int5":    return torch.uint5
        case "int6":    return torch.uint6
        case "int7":    return torch.uint7
        case "int8":    return torch.uint8
        case "int16":   return torch.uint16
        case "int32":   return torch.uint32
        case "int64":   return torch.uint64
        case "bool":    return torch.bool
        case _:         return None


def set_scheduler(args):
    global pipe
    if args.scheduler is not None:
        args.scheduler = json.loads(args.scheduler)
        pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)
    return


def load_lora(lora_dict, rank):
    global logger, pipe
    loras = json.loads(lora_dict)
    names = []

    for k, v in loras.items():
        logger.info(f"loading lora: {k}")
        if k.endswith(".safetensors") or k.endswith(".sft"):    weights = safetensors.torch.load_file(k, device=f'cuda:{rank}')
        else:                                                   weights = torch.load(k, map_location=torch.device(f'cuda:{rank}'))
        w = k.split("/")[-1]
        a = w if not "." in w else w.split(".")[0]
        names.append(a)
        pipe.load_lora_weights(weights, weight_name=w, adapter_name=a, local_files_only=True, low_cpu_mem_usage=True)
        del weights
        logger.info(f"Added LoRA (scale={v}): {k}")

    pipe.unet.set_adapters(names, list(loras.values()))
    loaded_adapters = pipe.unet.active_adapters()
    logger.info(f'Total loaded LoRAs: {len(loaded_adapters)}')
    logger.info(f'Adapters: {str(loaded_adapters)}')
    return names


def load_ip_adapter(ip_adapter_dict):
    global pipe
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


# TODO: implement
def load_pipeline(model_path, model_type, torch_dtype):
    if not model_path.endswith(".safetensors") and not model_path.endswith(".sft"):
        return None

    kwargs = []
    kwargs["torch_dtype"] = torch_dtype
    kwargs["use_safetensors"] = True
    kwargs["local_files_only"] = True
    kwargs["low_cpu_mem_usage"] = True

    match model_type:
        case "StableDiffusionPipeline":
            return StableDiffusionPipeline.from_single_file(model_path, **kwargs)
        case "StableDiffusion3Pipeline":
            return StableDiffusion3Pipeline.from_single_file(model_path, **kwargs)
        case "StableDiffusionXLPipeline":
            return StableDiffusionXLPipeline.from_single_file(model_path, **kwargs)
    return None


# TODO: implement
def load_model(model_path, config_path, model_type, torch_dtype):
    config = {}

    kwargs = {}
    kwargs["torch_dtype"] = torch_dtype
    kwargs["use_safetensors"] = True
    kwargs["local_files_only"] = True
    kwargs["low_cpu_mem_usage"] = True

    is_checkpoint = True
    if model_path.endswith(".safetensors") or model_path.endswith(".sft"):
        is_checkpoint = False
    elif model_path.endswith(".ckpt"):
        is_checkpoint = False
        kwargs["use_safetensors"] = False # NOTE: safetensors off
    elif model_path.endswith(".gguf"):
        is_checkpoint = False
        kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=torch_dtype)
        kwargs["use_safetensors"] = False # NOTE: safetensors off

    if not is_checkpoint:
        kwargs["config"] = config_path

    match model_type:
        case "AutoencoderKL":
            kwargs["torch_dtype"] = torch.float32
            if is_checkpoint:
                return AutoencoderKL.from_pretrained(model_path, **kwargs)
            else:
                return AutoencoderKL.from_single_file(model_path, **kwargs)
        case "ControlNetModel":
            if is_checkpoint:
                return ControlNetModel.from_pretrained(model_path, **kwargs)
            else:
                return ControlNetModel.from_single_file(model_path, **kwargs)
        case "FluxTransformer2DModel":
            if is_checkpoint:
                return FluxTransformer2DModel.from_pretrained(model_path, **kwargs)
            else:
                return FluxTransformer2DModel.from_single_file(model_path, **kwargs)
        case "MotionAdapter":
            if is_checkpoint:
                return MotionAdapter.from_pretrained(model_path, **kwargs)
            else:
                kwargs["config"] = f"{os.path.dirname(__file__)}/../resources/generic_motion_adapter_config.json"
                return MotionAdapter.from_single_file(model_path, **kwargs)
        case "SD3Transformer2DModel":
            if is_checkpoint:
                return SD3Transformer2DModel.from_pretrained(model_path, **kwargs)
            else:
                return SD3Transformer2DModel.from_single_file(model_path, **kwargs)
        case "UNet2DConditionModel":
            if is_checkpoint:
                return UNet2DConditionModel.from_pretrained(model_path, **kwargs)
            else:
                return UNet2DConditionModel.from_single_file(model_path, **kwargs)
        case "ZImageTransformer2DModel":
            if is_checkpoint:
                return ZImageTransformer2DModel.from_pretrained(model_path, **kwargs)
            else:
                return ZImageTransformer2DModel.from_single_file(model_path, **kwargs)
    return None


# TODO: implement
def setup_evals():
    global pipe

    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        pipe.transformer.eval()

    unet = getattr(pipe, "unet", None)
    if unet is not None:
        pipe.unet.eval()

    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is not None:
        pipe.text_encoder.eval()

    text_encoder_2 = getattr(pipe, "text_encoder_2", None)
    if text_encoder_2 is not None:
        pipe.text_encoder_2.eval()

    text_encoder_3 = getattr(pipe, "text_encoder_3", None)
    if text_encoder_3 is not None:
        pipe.text_encoder_3.eval()

    image_encoder = getattr(pipe, "image_encoder", None)
    if image_encoder is not None:
        pipe.image_encoder.eval()

    vae = getattr(pipe, "vae", None)
    if vae is not None:
        pipe.vae.eval()

    controlnet = getattr(pipe, "controlnet", None)
    if controlnet is not None:
        pipe.controlnet.eval()

    motion_adapter = getattr(pipe, "motion_adapter", None)
    if motion_adapter is not None:
        pipe.motion_adapter.eval()

    return


def get_quant_mapping(target, quantize_to):
    out = {}
    config = None
    if quantize_to.startswith("tao,"):
        quantize_to = quantize_to.replace("tao,", "")
        config = get_torchao_map(quantize_to)
    elif quantize_to.startswith("bnb,"):
        quantize_to = quantize_to.replace("bnb,", "")
        config = get_bnb_map(quantize_to)

    if config is not None:
        match target:
            case "unet":
                config = config[0]
                out["transformer"] = config
                out["unet"] = config
            case "encoder":
                config = config[1]
                out["text_encoder"] = config
                out["text_encoder_2"] = config
                out["text_encoder_3"] = config
                out["image_encoder"] = config
            case "vae":
                config = config[0]
                out["vae"] = config
            case "tokenizer":
                config = config[0]
                out["tokenizer"] = config
                out["tokenizer_2"] = config
            case "scheduler":
                config = config[0]
                out["scheduler"] = config
            case "misc":
                config = config[0]
                out["controlnet"] = config
                out["motion_adapter"] = config
    return out


def get_bnb_map(quantize_to):
    out = {}
    def get_bitsandbytes_config(t):
        c = {}
        if t.startswith("int8"):
            # int8
            c["load_in_8bit"] = True
            return c
        elif t.startswith("int4"):
            # int4,compute_type,quant_storage,quant_type
            t = t.split(",")
            if len(t) == 4:
                c["load_in_4bit"] = True
                c["bnb_4bit_compute_dtype"] = get_torch_type(t[1])
                c["bnb_4bit_quant_storage"] = get_torch_type(t[2])
                c["bnb_4bit_quant_type"] = t[3]
                return c
        return None

    t = get_bitsandbytes_config(quantize_to)
    if t is not None:
        return [BitsAndBytesConfigD(**t), BitsAndBytesConfigT(**t)]
    return None


def get_torchao_map(quantize_to):
    out = {}
    def get_torchao_config(t):
        if t == "int4wo":
            return "int4_weight_only"
        elif t == "int8wo":
            return "int8_weight_only"
        elif t.startswith("uint") and t.endswith("wo"):
            # uint1wo, uint2wo, uint3wo, uint4wo, uint5wo, uint6wo, uint7wo
            t = t.replace("uint", "").replace("wo", "")
            match t:
                case "1": t = torch.uint1
                case "2": t = torch.uint2
                case "3": t = torch.uint3
                case "4": t = torch.uint4
                case "5": t = torch.uint5
                case "6": t = torch.uint6
                case "7": t = torch.uint7
                case _: return None
            return UIntXWeightOnlyConfig(dtype=t)
        elif t == "float8wo":
            # float8wo
            return "float8_weight_only"
        elif t.startswith("fp") and "_" in t and "e" in t and "w" in t:
            # fpX_eAwB where X is the number of bits (1-7), A is exponent bits, and B is mantissa bits. Constraint: X == A + B + 1
            t = t.replace("fp", "").replace("_", "").replace("e", "").replace("w", "")
            try:
                x = int(t[0])
                a = int(t[1])
                b = int(t[2])
                if x == a + b + 1:
                    return FPXWeightOnlyConfig(ebits=a, mbits=b)
            except:
                return None
        return None

    t = get_torchao_config(quantize_to)
    if t is not None:
        return [TorchAoConfigD(t), TorchAoConfigT(t)]
    return None


def get_pipe_quant_config(mapping):
    return PipelineQuantizationConfig(quant_mapping=mapping)


def compile_helper(target, compile_config, adapter_names=None):
    global logger, pipe
    logger.info(f"compiling {target}")
    match target:
        case "transformer":
            if adapter_names is not None:
                pipe.transformer.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
                pipe.unload_lora_weights()
            pipe.transformer = torch.compile(pipe.transformer, **compile_config)
        case "unet":
            if adapter_names is not None:
                pipe.unet.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
                pipe.unload_lora_weights()
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

    logger.info(f"compiled {target}")
    return


# for xdit_usp only
def compile_model(logger):
    global pipe

    backend, mode, fullgraph = __get_compiler_config()
    logger.info(f"compiling model with {backend}:{mode}, fullgraph={fullgraph}")
    if len(mode) > 0:   pipe.model = torch.compile(pipe.model, backend=backend, mode=mode, fullgraph=fullgraph, dynamic=False)
    else:               pipe.model = torch.compile(pipe.model, backend=backend, fullgraph=fullgraph, dynamic=False)
    logger.info(f"compiled model")
    return


def print_params(data):
    global logger
    params = {
        "height":               data.get("height"),
        "width":                data.get("width"),
        "positive":             data.get("positive"),
        "negative":             data.get("negative"),
        "positive_embeds":      (data.get("positive_embeds") is not None),
        "negative_embeds":      (data.get("negative_embeds") is not None),
        "image":                (data.get("image") is not None),
        "ip_image":             (data.get("ip_image") is not None),
        "control_image":        (data.get("control_image") is not None),
        "latent":               (data.get("latent") is not None),
        "steps":                data.get("steps"),
        "cfg":                  data.get("cfg"),
        "controlnet_scale":     data.get("controlnet_scale"),
        "seed":                 data.get("seed"),
        "frames":               data.get("frames"),
        "decode_chunk_size":    data.get("decode_chunk_size"),
        "clip_skip":            data.get("clip_skip"),
        "motion_bucket_id":     data.get("motion_bucket_id"),
        "noise_aug_strength":   data.get("noise_aug_strength"),
        "sigmas":               (data.get("sigmas") is not None),
        "timesteps":            (data.get("timesteps") is not None),
        "denoising_start":      data.get("denoising_start"),
        "denoising_end":        data.get("denoising_end"),
    }
    formatted = "Received parameters:"
    for k, v in params.items():
        formatted += f'\n{k}:{str(v)}'
    logger.info(formatted)
    return


def print_mem_usage():
    global logger, pipe
    mem_usage_string = f'\n{"#" * 32}\n\nMemory usage:\n'
    for k, v in pipe.components.items():
        try:    mem = round(v.get_memory_footprint() / 1024 / 1024, 1)
        except: mem = "?"
        mem_usage_string += f"    {k}: {mem} MB\n"
    logger.info(f'{mem_usage_string}\n{"#" * 32}')
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


def process_input_latent(latents, dtype, device, timestep=None):
    global pipe

    latents = add_alpha_to_latent(latents)
    latents = latents.to(pipe.device)

    if timestep == None:    latents = pipe.scheduler.scale_model_input(latents, timestep=p.scheduler.timesteps[-1])
    else:                   latents = pipe.scheduler.scale_model_input(latents, timestep=timestep)
    latents = latents * pipe.vae.config.scaling_factor

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


def process_latent_for_output(latents, is_latent_output):
    global pipe
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


def convert_latent_to_output_latent(latents):
    latents = process_latent_for_output(latents, True)
    return latents


def convert_latent_to_image(latents):
    latents = process_latent_for_output(latents, False)
    return latents
