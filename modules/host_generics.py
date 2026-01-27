import gc
import inspect
import json
import logging
import numpy
import os
import safetensors
import torch
from diffusers import (
    AutoencoderKL,
    AutoModel,
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


local_rank = -1
def set_local_rank(rank):
    global local_rank
    local_rank = rank
    return

def get_local_rank():
    global local_rank
    assert local_rank > -1, "Distributed environment has not been initialized!"
    return local_rank


GENERIC_HOST_ARGS = {
    "port": int,
}


GENERIC_HOST_ARGS_TOGGLES = [
]


vae_fp16 = False
def set_vae_fp16(is_fp16):
    global vae_fp16
    vae_fp16 = is_fp16
    return


def get_vae_dtype():
    global vae_fp16
    if vae_fp16:    return torch.float16
    else:           return torch.float32


torch_dtype = torch.float16
def set_torch_dtype(dtype):
    global torch_dtype
    torch_dtype = dtype
    return


def get_torch_dtype():
    global torch_dtype
    return torch_dtype


def get_is_image_model(model):
    if model in ["ad", "svd", "wani2v"]:
        return False
    return True


initialized = False
def set_initialized(is_initialized):
    global initialized
    initialized = is_initialized
    if not is_initialized: clean()
    return


def get_initialized():
    global initialized
    return initialized


def get_initialized_flask():
    if get_initialized():   return "", 200
    else:                   return "", 202


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
    if pipe_in is None: clean()
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
def set_logger():
    global logger

    if logger is None:
        logging.root.setLevel(logging.NOTSET)
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(f"[Rank {str(get_local_rank())}]")
    return


def log(text, rank_0_only=False):
    global logger
    if rank_0_only == True and get_local_rank() != 0: return
    logger.info(text)
    return


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


def set_scheduler(scheduler_config):
    global pipe
    params = json.loads(scheduler_config)
    pipe.scheduler = get_scheduler(params, {})
    pipe.scheduler.set_timesteps(pipe.scheduler.config.num_train_timesteps)
    pipe.scheduler = get_scheduler(params, pipe.scheduler.config)
    return


def load_lora(loras):
    global logger, pipe
    # loras = json.loads(lora_dict)
    names = []

    for k, v in loras.items():
        log(f"loading lora: {k}")
        if k.endswith(".safetensors") or k.endswith(".sft"):    weights = safetensors.torch.load_file(k, device=f'cuda:{get_local_rank()}')
        else:                                                   weights = torch.load(k, map_location=torch.device(f'cuda:{get_local_rank()}'))
        w = k.split("/")[-1]
        a = w if not "." in w else w.split(".")[0]
        names.append(a)
        pipe.load_lora_weights(weights, weight_name=w, adapter_name=a, local_files_only=True, low_cpu_mem_usage=True)
        del weights
        log(f"Added LoRA (scale={v}): {k}")

    pipe.unet.set_adapters(names, list(loras.values()))
    loaded_adapters = pipe.unet.active_adapters()
    log(f'Total loaded LoRAs: {len(loaded_adapters)}')
    log(f'Adapters: {str(loaded_adapters)}')
    return names


def load_ip_adapter(ip_adapter):
    global pipe
    kwargs = {}
    split = ip_adapter.split("/")

    ip_adapter_file = split[-1]
    kwargs["weight_name"] = ip_adapter_file

    ip_adapter_subfolder = split[-2]
    kwargs["subfolder"] = ip_adapter_subfolder

    ip_adapter_folder = ip_adapter.replace(f'/{ip_adapter_subfolder}/{ip_adapter_file}', "")

    if "vit-h" in ip_adapter_file.lower():
        kwargs["image_encoder_folder"] = ip_adapter.replace(f'/{ip_adapter_subfolder}/{ip_adapter_file}', "/models/image_encoder")

    pipe.load_ip_adapter(
        ip_adapter_folder,
        use_safetensors=False, # NOTE: safetensors off
        local_files_only=True,
        low_cpu_mem_usage=True,
        **kwargs
    )
    return


def get_warmup_image():
    # image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
    image = load_image(f"{os.path.dirname(__file__)}/../resources/rocket.png") # 1024x576 pixels
    image = image.resize((768, 432), Image.Resampling.LANCZOS)
    return image


def __get_compiler_config(compiler_config):
    return config_compiler["backend"], config_compiler["mode"], config_compiler["fullgraph"]


# TODO: implement, required for pipeline .safetensors
def load_pipeline(model_path, model_type):
    global torch_dtype

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


def load_model(model_path, config_path, model_type):
    global torch_dtype
    log("Loading model: " + model_path + " with config: " + str(config_path))
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
        assert config_path is not None and len(config_path) > 0, "You must provide a config_path when loading from a single file"
        kwargs["config"] = config_path

    match model_type:
        case "AutoencoderKL":
            kwargs["torch_dtype"] = get_vae_dtype()
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
        case _:
            if is_checkpoint:
                return AutoModel.from_pretrained(model_path, **kwargs)
    return None


def setup_evals():
    global pipe
    for k, v in get_pipe().components.items():
        try: v.eval()
        except: pass
    return


def get_quant_mapping(target, quantize_to):
    out = {}
    config = None

    backend = quantize_to.pop("backend")
    match backend:
        case "bitsandbytes":
            load_in_8bit = quantize_to.get("load_in_8bit")
            load_in_4bit = quantize_to.get("load_in_4bit")
            assert not (load_in_8bit == True and load_in_4bit == True), "Select either 4-bit or 8-bit quantization, but not both"
            c = {}
            if load_in_8bit == True:
                for k,v in quantize_to.items():
                    if "4bit" in k: continue
                    c[k] = v
            elif load_in_4bit == True:
                for k,v in quantize_to.items():
                    if "int8" in k: continue
                    if k in ["bnb_4bit_compute_dtype", "bnb_4bit_quant_storage"]: v = get_torch_type(v)
                    c[k] = v
            else:
                return None
            log(c)
            config = [BitsAndBytesConfigD(**c), BitsAndBytesConfigT(**c)]
        case "torchao":
            def get_torchao_config(t):
                if t == "int4wo":
                    return "int4_weight_only"
                elif t == "int4dq":
                    return "int8_dynamic_activation_int4_weight"
                elif t == "int8wo":
                    return "int8_weight_only"
                elif t == "int8dq":
                    return "int8_dynamic_activation_int8_weight"
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
            quant_type = get_torchao_config(quantize_to["quant_type"])
            if quant_type is not None:  config = [TorchAoConfigD(quant_type), TorchAoConfigT(quant_type)]
            else:                       return None

    if config is not None:
        match target:
            case "unet":
                config = config[0]
                out["transformer"] = config
                out["transformer_2"] = config
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
            case "misc":
                config = config[0]
                out["controlnet"] = config
                out["motion_adapter"] = config
                out["image_processor"] = config
    return out


def get_pipe_quant_config(mapping):
    return PipelineQuantizationConfig(quant_mapping=mapping)


def compile_helper(target, compile_config, adapter_names=None):
    global logger, pipe
    log(f"compiling {target}")
    match target:
        case "transformer":
            if adapter_names is not None:
                pipe.transformer.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
                pipe.unload_lora_weights()
            pipe.transformer = torch.compile(pipe.transformer, **compile_config)
            if hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
                pipe.transformer_2 = torch.compile(pipe.text_encoder, **compile_config)
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
            log("unknown compile target - not compiling")
            return

    log(f"compiled {target}")
    return


def print_params(data):
    global logger
    formatted = "Received parameters:"
    for k, v in data.items():
        if torch.is_tensor(v) or len(str(v)) > 256:
            formatted += f'\n{k}:{str(v is not None)}'
        else:
            formatted += f'\n{k}:{str(v)}'
    log(formatted, rank_0_only=True)
    return


def print_mem_usage(with_devices=False):
    global logger, pipe
    mem_usage_string = f'\n{"#" * 32}\n\nMemory usage:\n'
    for k, v in pipe.components.items():
        try:    mem = round(v.get_memory_footprint() / 1024 / 1024, 1)
        except: mem = "?"
        mem_usage_string += f"    {k}: {mem} MB"
        if with_devices:
            try:    mem_usage_string += f" ({str(v.device)})"
            except: pass
        mem_usage_string += "\n"
    log(f'{mem_usage_string}\n{"#" * 32}', rank_0_only=True)
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


def set_scheduler_timesteps(start=0):
    global logger, pipe

    pipe_module = inspect.getmodule(pipe.__class__)
    if pipe_module is not None:
        if hasattr(pipe_module, 'old_retrieve_timesteps'):
            setattr(pipe_module, 'retrieve_timesteps', getattr(pipe_module, 'old_retrieve_timesteps'))
            delattr(pipe_module, 'old_retrieve_timesteps')
            if hasattr(pipe_module, 'new_retrieve_timesteps'):
                delattr(pipe_module, 'new_retrieve_timesteps')

        old_retrieve_timesteps = getattr(pipe_module, 'retrieve_timesteps', None)
        setattr(pipe_module, 'old_retrieve_timesteps', old_retrieve_timesteps)

        def new_retrieve_timesteps(*args, **kwargs):
            nonlocal start
            result = lambda: old_retrieve_timesteps(*args, **kwargs)
            timesteps, num_inference_steps = result()

            if len(timesteps) > 0 and start > 0:
                log("Old timesteps: " + str(timesteps), rank_0_only=True)
                timesteps = pipe.scheduler.timesteps[start * pipe.scheduler.order :]
                if hasattr(pipe.scheduler, "set_begin_index"):
                    pipe.scheduler.set_begin_index(start * pipe.scheduler.order)

                log("New timesteps: " + str(timesteps), rank_0_only=True)
            return timesteps, num_inference_steps - start
        setattr(pipe_module, 'retrieve_timesteps', new_retrieve_timesteps)
    return


def process_input_latent(latents):
    global logger, pipe, torch_dtype
    latents = add_alpha_to_latent(latents)
    latents = latents.to(device=pipe.device, dtype=torch_dtype)
    # latents = normalize_latent(latents, max_val=3)
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
    default_vae_dtype = pipe.vae.dtype
    latents = add_alpha_to_latent(latents)

    if is_latent_output:
        return latents

    pipe.vae = pipe.vae.to(dtype=get_vae_dtype())
    latents = latents.to(dtype=get_vae_dtype())
    latents = latents / pipe.vae.config.scaling_factor
    latents = pipe.vae.decode(latents, return_dict=False)[0]
    latents = pipe.image_processor.postprocess(latents, output_type="pil")
    pipe.vae = pipe.vae.to(dtype=default_vae_dtype)
    return latents


def convert_latent_to_output_latent(latents):
    latents = process_latent_for_output(latents, True)
    return latents


def convert_latent_to_image(latents):
    latents = process_latent_for_output(latents, False)
    return latents
