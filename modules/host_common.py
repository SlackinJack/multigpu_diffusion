import gc
import inspect
import json
import logging
import numpy
import os
import safetensors
import torch
import traceback
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
from diffusers import BitsAndBytesConfig as BitsAndBytesConfigD
from diffusers import QuantoConfig as QuantoConfigD
from diffusers import TorchAoConfig as TorchAoConfigD
from diffusers.hooks import apply_group_offloading
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_lora_to_diffusers
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file
from sdnq import SDNQConfig
from sdnq.common import use_torch_compile as triton_is_available
from torchao.quantization import (
    Float8WeightOnlyConfig,
    Float8DynamicActivationFloat8WeightConfig,
    FPXWeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
    UIntXWeightOnlyConfig,
)
from transformers import AutoModel as AutoModelT
from transformers import BitsAndBytesConfig as BitsAndBytesConfigT
from transformers import TorchAoConfig as TorchAoConfigT
from transformers import QuantoConfig as QuantoConfigT
from transformers import (
    AutoModelForCausalLM,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    Qwen3ForCausalLM,
    UMT5EncoderModel,
)
from PIL import Image


from compel import Compel, ReturnedEmbeddingsType
COMPEL_SUPPORTED_MODELS = ["sd1", "sd2", "sdxl"]


from modules.scheduler_config import *
from modules.utils import *


class HostShutdown(Exception): pass


GENERIC_HOST_ARGS = {
    "port": int,
}


GENERIC_HOST_ARGS_TOGGLES = [
]


def get_is_image_model(model):
    if model in ["ad", "svd", "wani2v", "want2v"]: return False
    return True


def clean():
    torch.cuda.memory.empty_cache()
    gc.collect()
    return


def setup_torch_dynamo(torch_config):
    cache_size_limit                = torch_config.get("torch_cache_limit")
    accumulated_cache_size_limit    = torch_config.get("torch_accumlated_cache_limit")
    capture_scalar_outputs          = torch_config.get("torch_capture_scalar")
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


def get_warmup_image():
    # image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
    image = load_image(f"{os.path.dirname(__file__)}/../resources/rocket.png") # 1024x576 pixels
    image = image.resize((768, 432), Image.Resampling.LANCZOS)
    return image


def __get_compiler_config(compiler_config):
    return config_compiler["backend"], config_compiler["mode"], config_compiler["fullgraph"]


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
                return out
            config = [BitsAndBytesConfigD(**c), BitsAndBytesConfigT(**c)]
        case "quanto":
            config = [QuantoConfigD(weights_dtype=quantize_to.get("quant_type")), QuantoConfigT(weights=quantize_to.get("quant_type"))]
        case "sdnq":
            # https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization
            config = [SDNQConfig(
                weights_dtype=quantize_to.get("quant_type"),
                group_size=0,                                   # higher: faster, lower quality, use 0 if triton_is_available
                svd_rank=16,                                    # higher: slower, higher quality (default: 32)
                svd_steps=4,                                    # higher: slower, higher quality (default: 8)
                dynamic_loss_threshold=1e-2,                    #
                use_svd=False,                                  # true: slower, higher quality (default: False)
                quant_conv=triton_is_available,                 # true: faster, lowers quality
                use_quantized_matmul=triton_is_available,       # true: faster
                use_quantized_matmul_conv=triton_is_available,  # true: faster
                dequantize_fp32=False,                          # true: slower, higher quality
                non_blocking=False,                             # ???
            )] * 2
        case "torchao":
            def get_torchao_config(t):
                # https://huggingface.co/docs/diffusers/main/quantization/torchao#supported-quantization-types
                if t == "int4wo":
                    return Int4WeightOnlyConfig()
                elif t == "int4dq":
                    return Int8DynamicActivationInt4WeightConfig()
                elif t == "int8wo":
                    return Int8WeightOnlyConfig()
                elif t == "int8dq":
                    return Int8DynamicActivationInt8WeightConfig()
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
                    return Float8WeightOnlyConfig()
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
            if quant_type is not None:  config = [TorchAoConfigD(quant_type), TorchAoConfigT(quant_type=quant_type)]

    if config is not None:
        match target:
            case "transformer":
                config = config[0]
                out["transformer"] = config
                out["transformer_2"] = config
                out["unet"] = config
            case "vae":
                config = config[0]
                out["vae"] = config
            case "misc":
                config = config[0]
                out["controlnet"] = config
                out["motion_adapter"] = config
                out["image_processor"] = config
            case "encoder":
                config = config[1]
                out["text_encoder"] = config
                out["text_encoder_2"] = config
                out["text_encoder_3"] = config
                out["image_encoder"] = config
            case "tokenizer":
                config = config[1]
                out["tokenizer"] = config
                out["tokenizer_2"] = config
    return out


def get_pipe_quant_config(mapping):
    return PipelineQuantizationConfig(quant_mapping=mapping)


def get_quantization_config(quantization_config):
    mappings = {}
    quantize_transformer                    = quantization_config.get("transformer")
    quantize_encoder                        = quantization_config.get("encoder")
    quantize_vae                            = quantization_config.get("vae")
    quantize_tokenizer                      = quantization_config.get("tokenizer")
    quantize_misc                           = quantization_config.get("misc")
    if quantize_transformer is not None:    mappings.update(get_quant_mapping("transformer", quantize_transformer))
    if quantize_encoder is not None:        mappings.update(get_quant_mapping("encoder", quantize_encoder))
    if quantize_vae is not None:            mappings.update(get_quant_mapping("vae", quantize_vae))
    if quantize_tokenizer is not None:      mappings.update(get_quant_mapping("tokenizer", quantize_tokenizer))
    if quantize_misc is not None:           mappings.update(get_quant_mapping("misc", quantize_misc))
    if len(list(mappings.keys())) > 0:
        return get_pipe_quant_config(mappings)


def normalize_latent(x, max_val=3.0):
    max_val = torch.tensor(max_val)
    x = x.detach().clone()
    for i in range(x.shape[0]):
        x[[i], :] = torch.sqrt(max_val**2 - 1) * x[[i], :] / torch.sqrt(torch.add(x[[i], :]**2, max_val**2 - 1))
        for chl in range(4):
            if x[i, chl, :, :].std() > 1.0:
                x[i, chl, :, :] /= x[i, chl, :, :].std()
    return x


def add_alpha_to_latent(latents):
    if latents.shape[1] == 3:
        alpha = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], device=latents.device)
        latents = torch.cat((latents, alpha), dim=1)
    return latents


def remove_alpha_from_latent(latents):
    if latents.shape[1] == 4:
        latents = latents[:, :3, :, :]
    return latents


class CommonHost:
    def __init__(self):
        self.local_rank = -1
        self.vae_dtype = False
        self.torch_dtype = torch.float16
        self.initialized = False
        self.progress = 0
        self.pipe = None
        self.pipeline_type = None
        self.logger = None
        self.default_scheduler = None
        self.adapter_names = None
        self.applied = None


    def get_initialized_flask(self):
        if self.initialized:    return "", 200
        else:                   return "", 202


    def get_progress_flask(self):
        return str(self.progress), 200


    def set_logger(self):
        if self.logger is None:
            logging.root.setLevel(logging.NOTSET)
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(f"[Rank {str(self.local_rank)}]")
        return


    def log(self, text, rank_0_only=False):
        if rank_0_only == True and self.local_rank != 0: return
        self.logger.info(text)
        return


    def set_scheduler(self, scheduler_config):
        params = json.loads(scheduler_config)
        self.pipe.scheduler = get_scheduler(params, {})
        self.pipe.scheduler.set_timesteps(self.pipe.scheduler.config.num_train_timesteps)
        self.pipe.scheduler = get_scheduler(params, self.pipe.scheduler.config)
        return


    def load_lora(self, loras):
        target = None
        if self.pipeline_type in ["sd1", "sd2", "sdxl"]:
            target = self.pipe.unet
        elif self.pipeline_type in ["sd3", "zimage"]:
            target = self.pipe.transformer

        if target is not None:
            names = []
            for k, v in loras.items():
                self.log(f"loading lora: {k}")
                if k.endswith(".safetensors") or k.endswith(".sft"):    weights = safetensors.torch.load_file(k, device=f'cuda:{self.local_rank}')
                else:                                                   weights = torch.load(k, map_location=torch.device(f'cuda:{self.local_rank}'))
                for k2,v2 in weights.items():                           weights[k2] = v2.to(device=target.device, dtype=target.dtype)
                w = k.split("/")[-1]
                a = w if not "." in w else w.split(".")[0]
                names.append(a)
                self.pipe.load_lora_weights(weights, weight_name=w, adapter_name=a, local_files_only=True, low_cpu_mem_usage=True)
                self.log(f"Added LoRA (scale={v}): {k}")

            target.set_adapters(names, list(loras.values()))
            loaded_adapters = target.active_adapters()
            self.log(f'Total loaded LoRAs: {len(loaded_adapters)}')
            self.log(f'Adapters: {str(loaded_adapters)}')
            if len(names) > 0:  self.adapter_names = names
            else:               self.adapter_names = None
        return


    def load_ip_adapter(self, ip_adapter):
        kwargs = {}
        split = ip_adapter.split("/")

        ip_adapter_file = split[-1]
        kwargs["weight_name"] = ip_adapter_file

        ip_adapter_subfolder = split[-2]
        kwargs["subfolder"] = ip_adapter_subfolder

        ip_adapter_folder = ip_adapter.replace(f'/{ip_adapter_subfolder}/{ip_adapter_file}', "")

        if "vit-h" in ip_adapter_file.lower():
            kwargs["image_encoder_folder"] = ip_adapter.replace(f'/{ip_adapter_subfolder}/{ip_adapter_file}', "/models/image_encoder")

        self.pipe.load_ip_adapter(
            ip_adapter_folder,
            use_safetensors=False, # NOTE: safetensors off
            local_files_only=True,
            low_cpu_mem_usage=True,
            **kwargs
        )
        return


    # TODO: implement for whole pipeline.safetensors
    """
    def load_pipeline(self, model_path, model_type):
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
    """


    def setup_pipeline(self, data, backend_name=None):
        # models
        backend_config                  = data.get("backend_config")
        pipeline_type                   = data.get("pipeline_type")
        variant                         = data.get("variant")
        checkpoint                      = data.get("checkpoint")
        transformer                     = data.get("transformer")
        vae                             = data.get("vae")
        vae_fp16                        = data.get("vae_fp16")
        control_net                     = data.get("control_net")
        text_encoder                    = data.get("text_encoder")
        text_encoder_2                  = data.get("text_encoder_2")
        text_encoder_3                  = data.get("text_encoder_3")
        motion_adapter                  = data.get("motion_adapter")
        motion_module                   = data.get("motion_module")
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
        enable_attention_slicing        = data.get("enable_attention_slicing")
        xformers_efficient              = data.get("xformers_efficient")
        group_offload_config            = data.get("group_offload_config")

        self.print_params(data)

        # checks
        assert not (pipeline_type == "ad" and motion_adapter is None and motion_module is None), "AnimateDiff requires providing a motion adapter/module."
        is_applied_config = True
        for k,v in data.items():
            if self.applied is None or k not in self.applied or v != self.applied[k]:
                is_applied_config = False
                break
        if is_applied_config:
            self.progress = 100
            return "", 200
        else:
            try:
                self.log(f"Initializing pipeline")

                # reset current
                self.progress = 0
                self.pipeline_type = pipeline_type
                if hasattr(self, "pipe"): del self.pipe
                clean()
                self.pipe = None
                PipelineClass = None

                # dynamo tweaks
                if torch_config is not None:
                    setup_torch_dynamo(torch_config)

                # torch tweaks
                setup_torch_backends()

                # update globals
                self.vae_dtype      = torch.float32 if (vae_fp16 is not None or vae_fp16 == False) else torch.float16
                self.torch_dtype    = get_torch_type(variant)

                kwargs = {}
                kwargs["torch_dtype"] = self.torch_dtype
                kwargs["use_safetensors"] = True
                kwargs["local_files_only"] = True
                kwargs["low_cpu_mem_usage"] = True
                kwargs["add_watermarker"] = False
                match backend_name:
                    case "balanced":
                        kwargs["device_map"] = "balanced"

                # quantize
                if quantization_config is not None:
                    kwargs["quantization_config"] = get_quantization_config(quantization_config)

                # set control net
                controlnet_model = None
                if control_net is not None and self.pipeline_type not in ["sdup", "svd"]:
                    kwargs["controlnet"] = self.load_model(control_net, "ControlNetModel")

                # set transformer
                if transformer is not None:
                    match self.pipeline_type:
                        case "flux":    kwargs["transformer"] = self.load_model(transformer, "FluxTransformer2DModel")
                        case "sd3":     kwargs["transformer"] = self.load_model(transformer, "SD3Transformer2DModel")
                        case "zimage":  kwargs["transformer"] = self.load_model(transformer, "ZImageTransformer2DModel")
                        case _:         kwargs["unet"] = self.load_model(transformer, "UNet2DConditionModel")

                # set vae
                if vae is not None and self.pipeline_type not in ["ad", "svd"]:
                    kwargs["vae"] = self.load_model(vae, "AutoencoderKL")

                # set text encoder(s)
                if text_encoder is not None or text_encoder_2 is not None or text_encoder_3 is not None:
                    if text_encoder is not None:
                        match self.pipeline_type:
                            case "sdxl":    kwargs["text_encoder"] = self.load_encoder(text_encoder, "CLIPTextModel")
                            case "sdup":    kwargs["text_encoder"] = self.load_encoder(text_encoder, "CLIPTextModel")
                            case "wani2v":  kwargs["text_encoder"] = self.load_encoder(text_encoder, "UMT5EncoderModel")
                            case "want2v":  kwargs["text_encoder"] = self.load_encoder(text_encoder, "UMT5EncoderModel")
                            case "zimage":  kwargs["text_encoder"] = self.load_encoder(text_encoder, "Qwen3ForCausalLM")
                    if text_encoder_2 is not None:
                        match self.pipeline_type:
                            case "sdxl":    kwargs["text_encoder_2"] = self.load_encoder(text_encoder_2, "CLIPTextModelWithProjection")
                    if text_encoder_3 is not None:
                        # kwargs["text_encoder_3"] = self.load_encoder(text_encoder_3, ???)
                        pass

                # set motion_adapter
                if (motion_module is not None or motion_adapter is not None) and self.pipeline_type in ["ad"]:
                    if motion_module is not None:
                        kwargs["motion_adapter"] = self.load_model(motion_module, "MotionAdapter")
                    else:
                        kwargs["motion_adapter"] = self.load_model(motion_adapter, "MotionAdapter")

                # setup pipeline
                match self.pipeline_type:
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

                self.pipe = PipelineClass.from_pretrained(checkpoint["checkpoint"], **kwargs)
                del kwargs
                self.default_scheduler = copy.deepcopy(self.pipe.scheduler)
                self.log("Pipeline initialized")
                self.progress = 50

                # for debugging
                if self.local_rank == 0:
                    # self.log("\n\n\n" + str(self.pipe.transformer) + "\n\n\n")
                    # raise ValueError
                    pass

                # set ipadapter
                if ip_adapter is not None:
                    self.load_ip_adapter(ip_adapter)

                # set memory saving
                if self.pipeline_type not in ["svd"]:
                    if enable_vae_slicing:          self.pipe.vae.enable_slicing()
                    if enable_vae_tiling:           self.pipe.vae.enable_tiling()
                    if enable_attention_slicing:    self.pipe.enable_attention_slicing()
                    if xformers_efficient:
                        if self.pipeline_type not in ["flux", "zimage"]:    self.pipe.enable_xformers_memory_efficient_attention()  # NOTE: blocked because causes tensor size mismatches
                        else:                                               self.log("xformers not supported for this pipeline - ignoring")

                # group offloading
                if group_offload_config is not None and backend_name not in ["balanced"]:
                    self.do_offloading(group_offload_config)
                else:
                    self.log("Group offloading not supported for this backend - ignoring")

                # set lora
                if lora is not None:
                    self.load_lora(lora)

                # set models to eval mode
                self.setup_evals()

                # compiles
                if compile_config is not None:
                    self.setup_compiles(compile_config)

                # clean up
                clean()

                # complete
                self.log("Model initialization completed")
                self.print_mem_usage()
                self.applied = data
                self.progress = 100

                return "", 200
            except:
                self.log(traceback.format_exc())
                return "", 500


    def load_model(self, model_dict, model_type):
        is_checkpoint = True
        config_path = None

        model_path = model_dict.get("checkpoint")
        if model_path is None:
            is_checkpoint = False
            model_path = model_dict.get("model")
            config_path = model_dict.get("config")
            assert config_path is not None, "You must provide a config_path when loading from a single file"

        self.log("Loading model: " + model_path + " with config: " + str(config_path))
        kwargs = {}
        kwargs["torch_dtype"] = self.torch_dtype
        kwargs["use_safetensors"] = True
        kwargs["local_files_only"] = True
        kwargs["low_cpu_mem_usage"] = True

        if model_path.endswith(".ckpt"):
            kwargs["use_safetensors"] = False # NOTE: safetensors off
        elif model_path.endswith(".gguf"):
            kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
            kwargs["use_safetensors"] = False # NOTE: safetensors off

        if not is_checkpoint:
            kwargs["config"] = config_path

        match model_type:
            case "AutoencoderKL":
                kwargs["torch_dtype"] = self.vae_dtype
                if is_checkpoint:   return AutoencoderKL.from_pretrained(model_path, **kwargs)
                else:               return AutoencoderKL.from_single_file(model_path, **kwargs)
            case "ControlNetModel":
                if is_checkpoint:   return ControlNetModel.from_pretrained(model_path, **kwargs)
                else:               return ControlNetModel.from_single_file(model_path, **kwargs)
            case "FluxTransformer2DModel":
                if is_checkpoint:   return FluxTransformer2DModel.from_pretrained(model_path, **kwargs)
                else:               return FluxTransformer2DModel.from_single_file(model_path, **kwargs)
            case "MotionAdapter":
                if is_checkpoint:   return MotionAdapter.from_pretrained(model_path, **kwargs)
                else:               return MotionAdapter.from_single_file(model_path, **kwargs)
            case "SD3Transformer2DModel":
                if is_checkpoint:   return SD3Transformer2DModel.from_pretrained(model_path, **kwargs)
                else:               return SD3Transformer2DModel.from_single_file(model_path, **kwargs)
            case "UNet2DConditionModel":
                if is_checkpoint:   return UNet2DConditionModel.from_pretrained(model_path, **kwargs)
                else:               return UNet2DConditionModel.from_single_file(model_path, **kwargs)
            case "ZImageTransformer2DModel":
                if is_checkpoint:   return ZImageTransformer2DModel.from_pretrained(model_path, **kwargs)
                else:               return ZImageTransformer2DModel.from_single_file(model_path, **kwargs)
            case _:
                if is_checkpoint:   return AutoModel.from_pretrained(model_path, **kwargs)
        return None


    def load_encoder(self, model_dict, model_type):
        is_checkpoint = True
        config_path = None

        model_path = model_dict.get("checkpoint")
        if model_path is None:
            is_checkpoint = False
            model_path = model_dict.get("model")
            config_path = model_dict.get("config")
            assert config_path is not None, "You must provide a config_path when loading from a single file"

        self.log("Loading model: " + model_path + " with config: " + str(config_path))
        kwargs = {}
        kwargs["torch_dtype"] = self.torch_dtype
        kwargs["local_files_only"] = True
        kwargs["low_cpu_mem_usage"] = True

        if not is_checkpoint:
            kwargs["config"] = config_path
            if model_path.endswith(".gguf"):
                kwargs["gguf_file"] = model_path
                model_path = model_path.split("/")[-1].split(".")[0]
            elif model_path.endswith(".safetensors") or model_path.endswith(".sft"):
                kwargs["state_dict"] = load_file(model_path)
                model_path = None

        match model_type:
            case "CLIPTextModel":
                return CLIPTextModel.from_pretrained(model_path, **kwargs)
            case "CLIPTextModelWithProjection":
                return CLIPTextModelWithProjection.from_pretrained(model_path, **kwargs)
            case "Qwen3ForCausalLM":
                return Qwen3ForCausalLM.from_pretrained(model_path, **kwargs)
            case "UMT5EncoderModel":
                return UMT5EncoderModel.from_pretrained(model_path, **kwargs)
            case _:
                return AutoModelT.from_pretrained(model_path, **kwargs)


    def setup_evals(self):
        for k, v in self.pipe.components.items():
            try: v.eval()
            except: pass
        return


    def do_offloading(self, group_offload_config):
        transformer_offload_config = group_offload_config.get("transformer")
        encoder_offload_config = group_offload_config.get("encoder")
        vae_offload_config = group_offload_config.get("vae")
        # TODO: misc_offload_config = group_offload_config.get("misc")
        kwargs = {"onload_device": torch.device(f"cuda:{self.local_rank}")}

        if transformer_offload_config is not None:
            kwargs["offload_device"] = transformer_offload_config.get("offload_device")
            kwargs["offload_type"] = transformer_offload_config.get("offload_type")
            if kwargs["offload_type"] == "block_level":
                kwargs["num_blocks_per_group"] = transformer_offload_config.get("num_blocks_per_group")
            if transformer_offload_config.get("use_stream") == True:
                kwargs["use_stream"] = True

            if hasattr(self.pipe, "transformer"):
                self.pipe.transformer.enable_group_offload(**kwargs)
            elif hasattr(self.pipe, "unet"):
                self.pipe.unet.enable_group_offload(**kwargs)

        if encoder_offload_config is not None:
            kwargs["offload_device"] = encoder_offload_config.get("offload_device")
            kwargs["offload_type"] = encoder_offload_config.get("offload_type")
            if kwargs["offload_type"] == "block_level":
                kwargs["num_blocks_per_group"] = encoder_offload_config.get("num_blocks_per_group")
            if encoder_offload_config.get("use_stream") == True:
                kwargs["use_stream"] = True
            if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
                apply_group_offloading(self.pipe.text_encoder, **kwargs)
            if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2 is not None:
                apply_group_offloading(self.pipe.text_encoder_2, **kwargs)
            if hasattr(self.pipe, "text_encoder_3") and self.pipe.text_encoder_3 is not None:
                apply_group_offloading(self.pipe.text_encoder_3, **kwargs)
            if hasattr(self.pipe, "image_encoder") and self.pipe.image_encoder is not None:
                apply_group_offloading(self.pipe.image_encoder, **kwargs)

        if vae_offload_config is not None:
            kwargs["offload_device"] = vae_offload_config.get("offload_device")
            kwargs["offload_type"] = vae_offload_config.get("offload_type")
            if kwargs["offload_type"] == "block_level":
                kwargs["num_blocks_per_group"] = vae_offload_config.get("num_blocks_per_group")
            if vae_offload_config.get("use_stream") == True:
                kwargs["use_stream"] = True

            if hasattr(self.pipe, "vae"):
                self.pipe.vae.enable_group_offload(**kwargs)


    def setup_compiles(self, compile_config):
        compile_transformer     = compile_config.get("compile_transformer")
        compile_vae             = compile_config.get("compile_vae")
        compile_encoder         = compile_config.get("compile_encoder")
        compile_backend         = compile_config.get("compile_backend")
        compile_mode            = compile_config.get("compile_mode")
        compile_options         = compile_config.get("compile_options")
        compile_fullgraph_off   = compile_config.get("compile_fullgraph_off")

        if compile_mode is not None and compile_options is not None:
            self.log("Compile mode and options are both defined, will ignore compile mode.")
            compile_mode = None
        compiler_config                             = {}
        compiler_config["fullgraph"]                = (compile_fullgraph_off is None or compile_fullgraph_off == False)
        compiler_config["dynamic"]                  = False
        if compile_backend is not None:             compiler_config["backend"] = compile_backend
        if compile_mode is not None:                compiler_config["mode"] = compile_mode
        if compile_options is not None:             compiler_config["options"] = json.loads(compile_options)

        if compile_transformer:
            if self.pipeline_type in ["flux", "sd3", "wani2v", "want2v", "zimage"]:
                self.compile_helper("transformer", compiler_config)
            else:
                self.compile_helper("unet", compiler_config)
        if compile_vae:                             self.compile_helper("vae", compiler_config)
        if compile_encoder:                         self.compile_helper("encoder", compiler_config)
        return


    def compile_helper(self, target, compile_config):
        self.log(f"compiling {target}")
        match target:
            case "transformer":
                if self.adapter_names is not None:
                    self.pipe.transformer.fuse_lora(adapter_names=self.adapter_names, lora_scale=1.0)
                    self.pipe.unload_lora_weights()
                self.pipe.transformer = torch.compile(self.pipe.transformer, **compile_config)
                if hasattr(self.pipe, "transformer_2") and self.pipe.transformer_2 is not None:
                    self.pipe.transformer_2 = torch.compile(self.pipe.text_encoder, **compile_config)
            case "unet":
                if self.adapter_names is not None:
                    self.pipe.unet.fuse_lora(adapter_names=self.adapter_names, lora_scale=1.0)
                    self.pipe.unload_lora_weights()
                self.pipe.unet = torch.compile(self.pipe.unet, **compile_config)
            case "vae":
                self.pipe.vae = torch.compile(self.pipe.vae, **compile_config)
            case "encoder":
                if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
                    self.pipe.text_encoder = torch.compile(self.pipe.text_encoder, **compile_config)
                if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2 is not None:
                    self.pipe.text_encoder_2 = torch.compile(self.pipe.text_encoder_2, **compile_config)
                if hasattr(self.pipe, "text_encoder_3") and self.pipe.text_encoder_3 is not None:
                    self.pipe.text_encoder_3 = torch.compile(self.pipe.text_encoder_3, **compile_config)
                if hasattr(self.pipe, "image_encoder") and self.pipe.image_encoder is not None:
                    self.pipe.image_encoder = torch.compile(self.pipe.image_encoder, **compile_config)
            case _:
                self.log("unknown compile target - not compiling")
                return

        self.log(f"compiled {target}")
        return


    def print_params(self, data):
        formatted = "Received parameters:"
        for k, v in data.items():
            if torch.is_tensor(v) or len(str(v)) > 256:
                formatted += f'\n{k}:{str(v is not None)}'
            else:
                formatted += f'\n{k}:{str(v)}'
        self.log(formatted, rank_0_only=True)
        return


    def prepare_inputs(self, data):
        height              = data.setdefault("height", None)
        width               = data.setdefault("width", None)
        positive            = data.setdefault("positive", None)
        negative            = data.setdefault("negative", None)
        positive_embeds     = data.setdefault("positive_embeds", None)
        negative_embeds     = data.setdefault("negative_embeds", None)
        image               = data.setdefault("image", None)
        ip_image            = data.setdefault("ip_image", None)
        control_image       = data.setdefault("control_image", None)
        latent              = data.setdefault("latent", None)
        steps               = data.setdefault("steps", None)
        cfg                 = data.setdefault("cfg", None)
        controlnet_scale    = data.setdefault("controlnet_scale", None)
        ip_adapter_scale    = data.setdefault("ip_adapter_scale", None)
        seed                = data.setdefault("seed", None)
        frames              = data.setdefault("frames", None)
        decode_chunk_size   = data.setdefault("decode_chunk_size", None)
        clip_skip           = data.setdefault("clip_skip", None)
        motion_bucket_id    = data.setdefault("motion_bucket_id", None)
        noise_aug_strength  = data.setdefault("noise_aug_strength", None)
        denoising_start     = data.setdefault("denoising_start", None)
        denoising_end       = data.setdefault("denoising_end", None)
        scheduler           = data.setdefault("scheduler", None)
        use_compel          = data.setdefault("use_compel", None)

        self.print_params(data)

        # checks
        if data["image"] is None and data["positive"] is None and data["positive_embeds"] is None:  return { "message": "No input provided", "output": None, "is_image": False }
        if data["positive"] is not None and data["positive_embeds"] is not None:                    return { "message": "Provide only one positive input", "output": None, "is_image": False }
        if data["negative"] is not None and data["negative_embeds"] is not None:                    return { "message": "Provide only one negative input", "output": None, "is_image": False }
        if data["image"] is None and self.pipeline_type in ["sdup", "svd", "wani2v"]:               return { "message": "No image provided for an image pipeline.", "output": None, "is_image": False }
        if data["ip_image"] is None and self.applied.get("ip_adapter") is not None:                 return { "message": "No IPAdapter image provided for a IPAdapter-loaded pipeline", "output": None, "is_image": False }
        if data["control_image"] is None and self.applied.get("control_net") is not None:           return { "message": "No ConstrolNet image provided for a ControlNet-loaded pipeline", "output": None, "is_image": False }

        data["image"]                                           = decode_b64_and_unpickle(image)
        data["ip_image"]                                        = decode_b64_and_unpickle(ip_image)
        data["control_image"]                                   = decode_b64_and_unpickle(control_image)
        data["latent"]                                          = decode_b64_and_unpickle(latent)
        data["positive_embeds"]                                 = decode_b64_and_unpickle(positive_embeds)
        data["negative_embeds"]                                 = decode_b64_and_unpickle(negative_embeds)
        if positive is not None and len(positive) == 0:         data["positive"] = None
        if negative is not None and len(negative) == 0:         data["negative"] = None
        if denoising_start is not None and denoising_start < 0: data["denoising_start"] = 0
        if denoising_end is not None and denoising_end > steps: data["denoising_end"]   = steps

        # load images
        if data["image"] is not None and self.pipeline_type in ["sdup", "svd", "wani2v"]:       data["image"] = load_image(data["image"])
        if data["ip_image"] is not None and self.applied.get("ip_adapter") is not None:         data["ip_image"] = load_image(data["ip_image"])
        if data["control_image"] is not None and self.applied.get("control_net") is not None:   data["control_image"] = load_image(data["control_image"])

        return data


    def print_mem_usage(self, with_devices=False):
        mem_usage_string = f'\n{"#" * 32}\n\nMemory usage:\n'
        for k, v in self.pipe.components.items():
            try:    mem = round(v.get_memory_footprint() / 1024 / 1024, 1)
            except: mem = "?"
            mem_usage_string += f"    {k}: {mem} MB"
            if with_devices:
                try:    mem_usage_string += f" ({str(v.device)})"
                except: pass
            mem_usage_string += "\n"
        self.log(f'{mem_usage_string}\n{"#" * 32}', rank_0_only=True)
        return


    def set_scheduler_timesteps(self, start=0):
        if start is not None and start > 0:
            pipe_module = inspect.getmodule(self.pipe.__class__)
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
                        self.log("Old timesteps: " + str(timesteps), rank_0_only=True)
                        timesteps = self.pipe.scheduler.timesteps[start * self.pipe.scheduler.order :]
                        if hasattr(self.pipe.scheduler, "set_begin_index"):
                            self.pipe.scheduler.set_begin_index(start * self.pipe.scheduler.order)

                        self.log("New timesteps: " + str(timesteps), rank_0_only=True)
                    return timesteps, num_inference_steps - start
                setattr(pipe_module, 'retrieve_timesteps', new_retrieve_timesteps)
            return


    def get_inference_kwargs(self, data, can_use_compel=True):
        # progress bar
        def set_step_progress(pipe, index, timestep, callback_kwargs):
            global get_scheduler_progressbar_offset_index
            nonlocal self, data
            the_index = get_scheduler_progressbar_offset_index(pipe.scheduler, index)
            self.log(str(callback_kwargs["latents"]), rank_0_only=True)
            self.progress = int(the_index / data["steps"] * 100)
            return callback_kwargs

        # compel
        data["positive_pooled_embeds"] = None
        data["negative_pooled_embeds"] = None
        if can_use_compel == True and data["use_compel"] == True and self.pipeline_type in COMPEL_SUPPORTED_MODELS and data["positive_embeds"] is None and data["negative_embeds"] is None:
            if self.pipeline_type in ["sd1", "sd2"]:    embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
            else:                                       embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
            compel = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                returned_embeddings_type=embeddings_type,
                requires_pooled=[False, True],
                truncate_long_prompts=False,
            )
            data["positive_embeds"], data["positive_pooled_embeds"] = compel([data["positive"]])
            if data["negative"] is not None and len(data["negative"]) > 0: data["negative_embeds"], data["negative_pooled_embeds"] = compel([data["negative"]])
            data["positive"] = data["negative"] = None
        else:
            if data["positive_embeds"] is not None:
                data["positive_pooled_embeds"]      = data["positive_embeds"][0][1]["pooled_output"]
                data["positive_embeds"]             = data["positive_embeds"][0][0]
            if data["negative_embeds"] is not None:
                data["negative_pooled_embeds"]      = data["negative_embeds"][0][1]["pooled_output"]
                data["negative_embeds"]             = data["negative_embeds"][0][0]

        # set pipe
        kwargs                                                  = {}
        kwargs["generator"]                                     = torch.Generator(device="cpu").manual_seed(data["seed"])
        kwargs["num_inference_steps"]                           = data["steps"]
        kwargs["callback_on_step_end"]                          = set_step_progress
        kwargs["callback_on_step_end_tensor_inputs"]            = ["latents"]
        match self.pipeline_type:
            case "ad":
                kwargs["output_type"]                           = "pil"
                kwargs["num_frames"]                            = data["frames"]
                kwargs["guidance_scale"]                        = data["cfg"]
                if data["ip_image"] is not None:                kwargs["ip_adapter_image"]      = data["ip_image"]
                if self.applied.get("control_net") is not None: kwargs["conditioning_frames"]   = [data["control_image"]] * data["frames"]
                if data["positive"] is not None:                kwargs["prompt"]                = data["positive"]
                if data["negative"] is not None:                kwargs["negative_prompt"]       = data["negative"]
                if data["height"] is not None:                  kwargs["height"]                = data["height"]
                if data["width"] is not None:                   kwargs["width"]                 = data["width"]
            case "sdup":
                kwargs["output_type"]                           = "pil"
                kwargs["guidance_scale"]                        = data["cfg"]
                if data["positive"] is not None:                kwargs["prompt"]                = data["positive"]
                if data["negative"] is not None:                kwargs["negative_prompt"]       = data["negative"]
                if data["image"] is not None:                   kwargs["image"]                 = data["image"]
            case "svd":
                kwargs["output_type"]                           = "pil"
                kwargs["num_frames"]                            = data["frames"]
                kwargs["decode_chunk_size"]                     = data["decode_chunk_size"]
                kwargs["motion_bucket_id"]                      = data["motion_bucket_id"]
                kwargs["noise_aug_strength"]                    = data["noise_aug_strength"]
                if data["image"] is not None:                   kwargs["image"]                 = data["image"]
                if data["height"] is not None:                  kwargs["height"]                = data["height"]
                if data["width"] is not None:                   kwargs["width"]                 = data["width"]
            case "want2v": # TODO: complete
                kwargs["output_type"]                           = "pil"
                kwargs["guidance_scale"]                        = data["cfg"]
                if data["height"] is not None:                  kwargs["height"]                = data["height"]
                if data["width"] is not None:                   kwargs["width"]                 = data["width"]
                if data["positive"] is not None:                kwargs["prompt"]                = data["positive"]
                if data["negative"] is not None:                kwargs["negative_prompt"]       = data["negative"]
                if data["frames"] is not None:                  kwargs["num_frames"]            = data["frames"]
            case "wani2v": # TODO: complete
                kwargs["output_type"]                           = "pil"
                kwargs["guidance_scale"]                        = data["cfg"]
                if data["image"] is not None:                   kwargs["image"]                 = data["image"]
                if data["height"] is not None:                  kwargs["height"]                = data["height"]
                if data["width"] is not None:                   kwargs["width"]                 = data["width"]
                if data["positive"] is not None:                kwargs["prompt"]                = data["positive"]
                if data["negative"] is not None:                kwargs["negative_prompt"]       = data["negative"]
                if data["frames"] is not None:                  kwargs["num_frames"]            = data["frames"]
            case "zimage": # TODO: complete
                kwargs["output_type"]                           = "latent"
                kwargs["guidance_scale"]                        = data["cfg"]
                if data["height"] is not None:                  kwargs["height"]                = data["height"]
                if data["width"] is not None:                   kwargs["width"]                 = data["width"]
                if data["positive"] is not None:                kwargs["prompt"]                = data["positive"]
                if data["negative"] is not None:                kwargs["negative_prompt"]       = data["negative"]
                if data["positive_embeds"] is not None:
                    kwargs["pooled_prompt_embeds"]              = data["positive_embeds"][0][1]["pooled_output"]
                    kwargs["prompt_embeds"]                     = data["positive_embeds"][0][0]
                if data["negative_embeds"] is not None:
                    kwargs["negative_pooled_embeds"]            = data["negative_embeds"][0][1]["pooled_output"]
                    kwargs["negative_embeds"]                   = data["negative_embeds"][0][0]
            case "flux": # TODO: complete
                kwargs["output_type"]                           = "latent"
                kwargs["guidance_scale"]                        = data["cfg"]
                if data["height"] is not None:                  kwargs["height"]                = data["height"]
                if data["width"] is not None:                   kwargs["width"]                 = data["width"]
                if data["positive"] is not None:                kwargs["prompt"]                = data["positive"]
                if data["negative"] is not None:                kwargs["negative_prompt"]       = data["negative"]
                if data["positive_embeds"] is not None:
                    kwargs["pooled_prompt_embeds"]              = data["positive_embeds"][0][1]["pooled_output"]
                    kwargs["prompt_embeds"]                     = data["positive_embeds"][0][0]
                if data["negative_embeds"] is not None:
                    kwargs["negative_pooled_embeds"]            = data["negative_embeds"][0][1]["pooled_output"]
                    kwargs["negative_embeds"]                   = data["negative_embeds"][0][0]
            case _: # NOTE: "sd1", "sd2", "sd3", "sdxl"
                kwargs["output_type"]                           = "latent"
                kwargs["guidance_scale"]                        = data["cfg"]
                if self.pipeline_type in ["sd1", "sd2", "sd3", "sdxl"]: kwargs["clip_skip"] = data["clip_skip"]

                if data["latent"] is not None:
                    kwargs["latents"] = data["latent"]
                else:
                    if data["height"] is not None:              kwargs["height"]                    = data["height"]
                    if data["width"] is not None:               kwargs["width"]                     = data["width"]
                if data["positive"] is not None:                kwargs["prompt"]                    = data["positive"]
                if data["negative"] is not None:                kwargs["negative_prompt"]           = data["negative"]
                if data["positive_embeds"] is not None:         kwargs["prompt_embeds"]             = data["positive_embeds"]
                if data["positive_pooled_embeds"] is not None:  kwargs["pooled_prompt_embeds"]      = data["positive_pooled_embeds"]
                if data["negative_embeds"] is not None:         kwargs["negative_embeds"]           = data["negative_embeds"]
                if data["negative_pooled_embeds"] is not None:  kwargs["negative_pooled_embeds"]    = data["negative_pooled_embeds"]
                if data["denoising_end"] is not None:           kwargs["denoising_end"]             = float(data["denoising_end"] / data["steps"])

                if self.applied.get("ip_adapter") is not None and data["ip_image"] is not None:
                    kwargs["ip_adapter_image"]                  = data["ip_image"]
                    if data["ip_adapter_scale"] is not None:    self.pipe.set_ip_adapter_scale(data["ip_adapter_scale"])
                    else:                                       self.pipe.set_ip_adapter_scale(1.0)
                if self.applied.get("control_net") is not None and data["control_image"] is not None:
                    kwargs["image"]                             = data["control_image"]
                    if data["controlnet_scale"] is not None:    kwargs["controlnet_conditioning_scale"] = data["controlnet_scale"]
                    else:                                       kwargs["controlnet_conditioning_scale"] = 1.0
        return kwargs


    def process_input_latent(self, latents):
        latents = add_alpha_to_latent(latents)
        latents = latents.to(device=self.pipe.device, dtype=self.torch_dtype)
        # latents = normalize_latent(latents, max_val=3)
        return latents


    def process_latent_for_output(self, latents, is_latent_output):
        default_vae_dtype = self.pipe.vae.dtype
        latents = add_alpha_to_latent(latents)

        if is_latent_output:
            return latents

        self.pipe.vae = self.pipe.vae.to(dtype=self.vae_dtype)
        latents = latents.to(dtype=self.vae_dtype)
        latents = latents / self.pipe.vae.config.scaling_factor
        latents = self.pipe.vae.decode(latents, return_dict=False)[0]
        latents = self.pipe.image_processor.postprocess(latents, output_type="pil")
        self.pipe.vae = self.pipe.vae.to(dtype=default_vae_dtype)
        return latents


    def convert_latent_to_output_latent(self, latents):
        latents = self.process_latent_for_output(latents, True)
        return latents


    def convert_latent_to_image(self, latents):
        latents = self.process_latent_for_output(latents, False)
        return latents
