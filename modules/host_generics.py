import gc
import json
import logging
import mmap
import os
import safetensors
import torch
import torch._dynamo
from optimum.quanto import freeze, qfloat8, qint2, qint4, qint8, quantize


config          = json.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../config.json"))
config_compiler = config["compiler"]


GENERIC_HOST_ARGS = {
    "height":               int,
    "width":                int,
    "warm_up_steps":        int,
    "port":                 int,
    "type":                 str,
    "variant":              str,
    "scheduler":            str,
    "quantize_to":          str,
    "model":                str,    # path
    "gguf_model":           str,    # path
    "motion_module":        str,    # path
    "motion_adapter":       str,    # path
    "motion_adapter_lora":  str,    # path
    "control_net":          str,    # path
    "vae":                  str,    # path
    "lora":                 str,    # json dict > { "path": scale, ... }
    "ip_adapter":           str,    # json dict > { "path": scale, ... }
    "image_scale":          float,
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
    #torch._dynamo.config.suppress_errors = True
    #torch._dynamo.config.capture_scalar_outputs = False
    torch._dynamo.config.cache_size_limit = int(8 * 4)
    torch._dynamo.config.accumulated_cache_size_limit = int(64 * 4)
    return


def setup_torch_backends():
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


def __get_compiler_config():
    global config_compiler
    return config_compiler["backend"], config_compiler["mode"], config_compiler["fullgraph"]


def compile_unet(pipe, adapter_names, logger, is_distrifuser=False):
    backend, mode, fullgraph = __get_compiler_config()
    logger.info(f"compiling unet with {backend}:{mode}, fullgraph={fullgraph}")
    if adapter_names:
        if is_distrifuser:  pipe.unet.model.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
        else:               pipe.unet.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
        pipe.unload_lora_weights()
    if is_distrifuser:
        if len(mode) > 0:   pipe.unet.model = torch.compile(pipe.unet.model, backend=backend, mode=mode, fullgraph=fullgraph, dynamic=False)
        else:               pipe.unet.model = torch.compile(pipe.unet.model, backend=backend, fullgraph=fullgraph, dynamic=False)
    else:
        if len(mode) > 0:   pipe.unet = torch.compile(pipe.unet, backend=backend, mode=mode, fullgraph=fullgraph, dynamic=False)
        else:               pipe.unet = torch.compile(pipe.unet, backend=backend, fullgraph=fullgraph, dynamic=False)
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

