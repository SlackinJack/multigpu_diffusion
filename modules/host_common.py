import gc
import json
import logging
import numpy
import os
import safetensors
import torch
import torch.nn.functional as F
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
    WanImageToVideoPipeline,
    WanPipeline,
    ZImagePipeline,
    # ZImageControlNetPipeline,
)
from diffusers import BitsAndBytesConfig as BitsAndBytesConfigD
from diffusers import QuantoConfig as QuantoConfigD
from diffusers import TorchAoConfig as TorchAoConfigD
from diffusers.hooks import apply_group_offloading
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
from modules.utils import (
    clean,
    clean_override_function,
    get_function_from_class,
    override_function,
    normalize_latent,
    add_alpha_to_latent,
    remove_alpha_from_latent,
    get_torch_type,
    decode_b64_and_unpickle,
    pickle_and_encode_b64,
    convert_b64_to_nhwc_tensor,
    convert_image_to_hwc_tensor,
    convert_tensor_to_b64,
    format_json
)


GENERIC_HOST_ARGS = {
    "port": int,
}


GENERIC_HOST_ARGS_TOGGLES = [
]


# these pipelines do not natively support denoising_start (uses a custom workaround in callback_on_step_end):
DENOISING_START_WORKAROUND_PIPELINES = ["zimage"]

# these pipelines do not support using torch deterministic algorithms for whatever reason (no workarounds):
NON_DETERMINISTIC_PIPELINES = ["zimage"]


def _get_transformer_module_names():
    return ["transformer", "transformer_2", "unet"]


def _get_encoder_module_names():
    return ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]


def _get_misc_module_names():
    return ["controlnet", "motion_adapter", "image_processor", "feature_extractor"]


def _get_vae_module_names():
    return ["vae"]


def _get_tokenizer_module_names():
    return ["tokenizer", "tokenizer_2"]


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
            kwargs = { "weights_dtype": quantize_to.pop("quant_type") }
            for k,v in quantize_to.items(): kwargs[k] = v
            config = [SDNQConfig(**kwargs)] * 2
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
                for name in _get_transformer_module_names():
                    out[name] = config[0]
            case "vae":
                for name in _get_vae_module_names():
                    out[name] = config[0]
            case "misc":
                for name in _get_misc_module_names():
                    out[name] = config[0]
            case "encoder":
                for name in _get_encoder_module_names():
                    out[name] = config[1]
            case "tokenizer":
                for name in _get_tokenizer_module_names():
                    out[name] = config[1]
    return out


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
        return PipelineQuantizationConfig(quant_mapping=mappings)


class CommonHost:
    def __init__(self):
        self.local_rank = -1
        self.vae_dtype = None
        self.torch_dtype = torch.float16
        self.initialized = False
        self.progress = 0
        self.pipe = None
        self.pipeline_type = None
        self.logger = None
        self.default_scheduler = None
        self.adapter_names = None
        self.applied = None
        self.is_image_model = True
        self.is_transformer_model_type = False
        self.can_use_deepcache = False
        self.can_use_cachedit = False


    def get_initialized_flask(self):
        if self.initialized:    return "", 200
        else:                   return "", 202


    def get_progress_flask(self):
        return str(self.progress), 200


    def set_logger(self):
        if self.logger is None:
            logger = logging.getLogger(name=str(self.local_rank))
            logger.setLevel(logging.INFO)
            if logger.hasHandlers():
                logger.handlers.clear()
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(fmt=f'[Rank {str(self.local_rank)}]: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
            self.logger = logger
        return


    def log(self, text, rank_0_only=True):
        if rank_0_only == True and self.local_rank != 0: return
        self.logger.info(text)
        # print(f"[Rank {str(self.local_rank)}]: {text}")
        return


    def get_applied(self):
        return str(self.applied), 200


    def close_pipeline(self):
        # TODO: fix this method to properly free up resources
        # self.local_rank = -1
        self.vae_dtype = None
        self.torch_dtype = torch.float16
        self.initialized = False
        self.progress = 0
        self.pipe = None
        self.pipeline_type = None
        # self.logger = None
        self.default_scheduler = None
        self.adapter_names = None
        self.applied = None
        self.is_image_model = True
        self.is_transformer_model_type = False
        self.can_use_deepcache = False
        self.can_use_cachedit = False
        return


    def set_scheduler(self, scheduler_config):
        params = json.loads(scheduler_config)
        self.pipe.scheduler = get_scheduler(params)
        return


    def setup_pipeline(self, data, backend_name=None):
        # models
        data["backend_config"]              = data.setdefault("backend_config")
        data["pipeline_type"]               = data.setdefault("pipeline_type")
        data["variant"]                     = data.setdefault("variant")
        data["checkpoint"]                  = data.setdefault("checkpoint")
        data["transformer"]                 = data.setdefault("transformer")
        data["vae"]                         = data.setdefault("vae")
        data["vae_fp16"]                    = data.setdefault("vae_fp16")
        data["control_net"]                 = data.setdefault("control_net")
        data["text_encoder"]                = data.setdefault("text_encoder")
        data["text_encoder_2"]              = data.setdefault("text_encoder_2")
        data["text_encoder_3"]              = data.setdefault("text_encoder_3")
        data["motion_adapter"]              = data.setdefault("motion_adapter")
        data["motion_module"]               = data.setdefault("motion_module")
        data["ip_adapter"]                  = data.setdefault("ip_adapter")
        data["lora"]                        = data.setdefault("lora")

        # memory & optimizations
        data["enable_vae_slicing"]          = data.setdefault("enable_vae_slicing")
        data["enable_vae_tiling"]           = data.setdefault("enable_vae_tiling")
        data["enable_attention_slicing"]    = data.setdefault("enable_attention_slicing")
        data["xformers_efficient"]          = data.setdefault("xformers_efficient")
        data["group_offload_config"]        = data.setdefault("group_offload_config")
        data["sd_fuse_qkv_projections"]     = data.setdefault("sd_fuse_qkv_projections")
        # compile
        data["compile_config"]              = data.setdefault("compile_config")
        data["torch_config"]                = data.setdefault("torch_config")
        # quantization
        data["quantization_config"]         = data.setdefault("quantization_config")
        # attention
        data["attn_backend_config"]         = data.setdefault("attn_backend_config")

        self.print_params(data)

        # checks
        assert not (data["pipeline_type"] == "ad" and motion_adapter is None and data["motion_module"] is None), "AnimateDiff requires providing a motion adapter/module."
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
                self.log(f"⚙️ Initializing pipeline", rank_0_only=False)

                # reset current
                self.close_pipeline()
                clean()

                # setup current
                self.pipeline_type = data["pipeline_type"]

                # torch tweaks
                if data["torch_config"] is not None:
                    cache_size_limit                = data["torch_config"].get("torch_cache_limit")
                    accumulated_cache_size_limit    = data["torch_config"].get("torch_accumlated_cache_limit")
                    capture_scalar_outputs          = data["torch_config"].get("torch_capture_scalar")
                    if cache_size_limit is not None:                torch._dynamo.config.cache_size_limit               = cache_size_limit
                    if accumulated_cache_size_limit is not None:    torch._dynamo.config.accumulated_cache_size_limit   = accumulated_cache_size_limit
                    if capture_scalar_outputs is not None:          torch._dynamo.config.capture_scalar_outputs         = capture_scalar_outputs
                torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True, enable_cudnn=True)
                torch._inductor.config.conv_1x1_as_mm = True
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.epilogue_fusion = False
                torch._inductor.config.coordinate_descent_check_all_directions = True
                torch._inductor.config.force_fuse_int_mm_with_mul = True
                torch._inductor.config.use_mixed_mm = True
                if self.pipeline_type not in NON_DETERMINISTIC_PIPELINES:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)

                # update globals
                self.vae_dtype      = torch.float16 if data["vae_fp16"] == True else None
                self.torch_dtype    = get_torch_type(data["variant"])

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
                if data["quantization_config"] is not None:
                    kwargs["quantization_config"] = get_quantization_config(data["quantization_config"])

                # set transformer
                if data["transformer"] is not None:
                    match self.pipeline_type:
                        case "flux":    kwargs["transformer"] = self.load_model(data["transformer"], "FluxTransformer2DModel")
                        case "sd3":     kwargs["transformer"] = self.load_model(data["transformer"], "SD3Transformer2DModel")
                        case "zimage":  kwargs["transformer"] = self.load_model(data["transformer"], "ZImageTransformer2DModel")
                        case _:         kwargs["unet"] = self.load_model(data["transformer"], "UNet2DConditionModel")
                self.progress = 10

                # set vae
                if data["vae"] is not None and self.pipeline_type not in ["ad", "svd"]:
                    kwargs["vae"] = self.load_model(data["vae"], "AutoencoderKL")
                self.progress = 20

                # set text encoder(s)
                if data["text_encoder"] is not None or data["text_encoder_2"] is not None or data["text_encoder_3"] is not None:
                    if data["text_encoder"] is not None:
                        match self.pipeline_type:
                            case "sdxl":    kwargs["text_encoder"] = self.load_encoder(data["text_encoder"], "CLIPTextModel")
                            case "sdup":    kwargs["text_encoder"] = self.load_encoder(data["text_encoder"], "CLIPTextModel")
                            case "wani2v":  kwargs["text_encoder"] = self.load_encoder(data["text_encoder"], "UMT5EncoderModel")
                            case "want2v":  kwargs["text_encoder"] = self.load_encoder(data["text_encoder"], "UMT5EncoderModel")
                            case "zimage":  kwargs["text_encoder"] = self.load_encoder(data["text_encoder"], "Qwen3ForCausalLM")
                    if data["text_encoder_2"] is not None:
                        match self.pipeline_type:
                            case "sdxl":    kwargs["text_encoder_2"] = self.load_encoder(data["text_encoder_2"], "CLIPTextModelWithProjection")
                    if data["text_encoder_3"] is not None:
                        # TODO: complete
                        # kwargs["text_encoder_3"] = self.load_encoder(data["text_encoder_3"], ???)
                        pass
                self.progress = 30

                # set control net
                controlnet_model = None
                if data["control_net"] is not None and self.pipeline_type not in ["sdup", "svd"]:
                    kwargs["controlnet"] = self.load_model(data["control_net"], "ControlNetModel")

                # set motion_adapter
                if (data["motion_module"] is not None or data["motion_adapter"] is not None) and self.pipeline_type in ["ad"]:
                    if data["motion_module"] is not None:
                        kwargs["motion_adapter"] = self.load_model(data["motion_module"], "MotionAdapter")
                    else:
                        kwargs["motion_adapter"] = self.load_model(data["motion_adapter"], "MotionAdapter")
                self.progress = 40

                # setup pipeline
                match self.pipeline_type:
                    case "ad":
                        PipelineClass = AnimateDiffControlNetPipeline if data["control_net"] is not None else AnimateDiffPipeline
                        self.is_image_model = False
                        self.can_use_deepcache = True # TODO: test
                    case "flux":
                        PipelineClass = FluxControlNetPipeline if data["control_net"] is not None else FluxPipeline
                        self.is_transformer_model_type = True
                        self.can_use_cachedit = True
                    case "sd1":
                        PipelineClass = StableDiffusionControlNetPipeline if data["control_net"] is not None else StableDiffusionPipeline
                        self.can_use_deepcache = True
                    case "sd2":
                        PipelineClass = StableDiffusionControlNetPipeline if data["control_net"] is not None else StableDiffusionPipeline
                        self.can_use_deepcache = True
                    case "sd3":
                        PipelineClass = StableDiffusion3ControlNetPipeline if data["control_net"] is not None else StableDiffusion3Pipeline
                        self.is_transformer_model_type = True
                        self.can_use_cachedit = True
                    case "sdup":
                        PipelineClass = StableDiffusionUpscalePipeline
                        self.can_use_deepcache = True # TODO: test
                    case "sdxl":
                        PipelineClass = StableDiffusionXLControlNetPipeline if data["control_net"] is not None else StableDiffusionXLPipeline
                        self.can_use_deepcache = True
                    case "svd":
                        PipelineClass = StableVideoDiffusionPipeline
                        self.is_image_model = False
                        self.can_use_deepcache = True # TODO: test
                    case "want2v":
                        PipelineClass = WanPipeline
                        self.is_image_model = False
                        self.is_transformer_model_type = True
                        self.can_use_cachedit = True
                    case "wani2v":
                        PipelineClass = WanImageToVideoPipeline
                        self.is_image_model = False
                        self.is_transformer_model_type = True
                        self.can_use_cachedit = True # TODO: test
                    case "zimage":
                        # PipelineClass = ZImageControlNetPipeline if data["control_net"] is not None else ZImagePipeline
                        PipelineClass = ZImagePipeline
                        self.is_transformer_model_type = True
                        self.can_use_cachedit = True
                    case _: raise NotImplementedError
                self.pipe = PipelineClass.from_pretrained(data["checkpoint"]["checkpoint"], **kwargs)
                del kwargs
                self.default_scheduler = copy.deepcopy(self.pipe.scheduler)
                if data["attn_backend_config"] is not None and data["attn_backend_config"].get("backend") is not None:
                    if self.is_transformer_model_type == True:  self.pipe.transformer.set_attention_backend(data["attn_backend_config"]["backend"])
                    else:                                       self.pipe.unet.set_attention_backend(data["attn_backend_config"]["backend"])
                    self.log(f"ℹ️ Set attention backend to {data["attn_backend_config"]["backend"]}")
                self.log("✅ Pipeline initialized", rank_0_only=False)
                self.progress = 50

                # for debugging
                if self.local_rank == 0:
                    # self.log("\n\n\n" + str(self.pipe.transformer) + "\n\n\n")
                    # raise ValueError
                    pass

                # set ipadapter
                if data["ip_adapter"] is not None:
                    kwargs = {}
                    ip_adapter_model = data["ip_adapter"].get("model")
                    split = ip_adapter_model.split("/")

                    ip_adapter_file = split[-1]
                    kwargs["weight_name"] = ip_adapter_file

                    ip_adapter_subfolder = split[-2]
                    kwargs["subfolder"] = ip_adapter_subfolder

                    ip_adapter_folder = ip_adapter_model.replace(f'/{ip_adapter_subfolder}/{ip_adapter_file}', "")

                    if "vit-h" in ip_adapter_file.lower():
                        kwargs["image_encoder_folder"] = ip_adapter_model.replace(f'/{ip_adapter_subfolder}/{ip_adapter_file}', "/models/image_encoder")

                    self.pipe.load_ip_adapter(
                        ip_adapter_folder,
                        use_safetensors=False, # NOTE: safetensors off
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        **kwargs
                    )
                self.progress = 60

                # set memory saving
                if self.pipeline_type not in ["svd"]:
                    if data["enable_vae_slicing"] == True:          self.pipe.vae.enable_slicing()
                    if data["enable_vae_tiling"] == True:           self.pipe.vae.enable_tiling()
                    if data["enable_attention_slicing"] == True:    self.pipe.enable_attention_slicing()
                    if data["xformers_efficient"] == True:
                        if self.pipeline_type not in ["flux", "zimage"]:    self.pipe.enable_xformers_memory_efficient_attention()  # NOTE: blocked because causes tensor size mismatches
                        else:                                               self.log("⚠️ xformers not supported for this pipeline - ignoring")
                self.progress = 70

                # group offloading
                if data["group_offload_config"] is None:
                    pass
                else:
                    if backend_name in ["balanced"]:
                        self.log("❌ Group offloading not supported for this host - ignoring")
                    else:
                        transformer_offload_config = data["group_offload_config"].get("transformer")
                        encoder_offload_config = data["group_offload_config"].get("encoder")
                        vae_offload_config = data["group_offload_config"].get("vae")
                        misc_offload_config = data["group_offload_config"].get("misc")
                        kwargs = {"onload_device": torch.device(f"cuda:{self.local_rank}")}

                        if transformer_offload_config is not None:
                            kwargs["offload_device"] = transformer_offload_config.get("offload_device")
                            kwargs["offload_type"] = transformer_offload_config.get("offload_type")
                            if kwargs["offload_type"] == "block_level":
                                kwargs["num_blocks_per_group"] = transformer_offload_config.get("num_blocks_per_group")
                            if transformer_offload_config.get("use_stream") == True:
                                kwargs["use_stream"] = True

                            for name in _get_transformer_module_names():
                                if hasattr(self.pipe, name):
                                    module = getattr(self.pipe, name)
                                    if module is not None:
                                        module.enable_group_offload(**kwargs)

                        if encoder_offload_config is not None:
                            kwargs["offload_device"] = encoder_offload_config.get("offload_device")
                            kwargs["offload_type"] = encoder_offload_config.get("offload_type")
                            if kwargs["offload_type"] == "block_level":
                                kwargs["num_blocks_per_group"] = encoder_offload_config.get("num_blocks_per_group")
                            if encoder_offload_config.get("use_stream") == True:
                                kwargs["use_stream"] = True
                            for name in _get_encoder_module_names():
                                if hasattr(self.pipe, name):
                                    module = getattr(self.pipe, name)
                                    if module is not None:
                                        apply_group_offloading(module, **kwargs)

                        if vae_offload_config is not None:
                            kwargs["offload_device"] = vae_offload_config.get("offload_device")
                            kwargs["offload_type"] = vae_offload_config.get("offload_type")
                            if kwargs["offload_type"] == "block_level":
                                kwargs["num_blocks_per_group"] = vae_offload_config.get("num_blocks_per_group")
                            if vae_offload_config.get("use_stream") == True:
                                kwargs["use_stream"] = True
                            for name in _get_vae_module_names():
                                if hasattr(self.pipe, name):
                                    module = getattr(self.pipe, name)
                                    if module is not None:
                                        module.enable_group_offload(**kwargs)

                        if misc_offload_config is not None:
                            kwargs["offload_device"] = misc_offload_config.get("offload_device")
                            kwargs["offload_type"] = misc_offload_config.get("offload_type")
                            if kwargs["offload_type"] == "block_level":
                                kwargs["num_blocks_per_group"] = misc_offload_config.get("num_blocks_per_group")
                            if misc_offload_config.get("use_stream") == True:
                                kwargs["use_stream"] = True
                            for name in _get_misc_module_names():
                                if hasattr(self.pipe, name):
                                    module = getattr(self.pipe, name)
                                    if module is not None:
                                        try: module.enable_group_offload(**kwargs)
                                        except: pass
                self.progress = 80

                # set lora
                if data["lora"] is not None:
                    target = None
                    if self.is_transformer_model_type == True:  target = self.pipe.transformer
                    else:                                       target = self.pipe.unet

                    if target is not None:
                        names = []
                        for k, v in data["lora"].items():
                            self.log(f"⏳ Loading lora: {k}")
                            if k.endswith(".safetensors") or k.endswith(".sft"):    weights = safetensors.torch.load_file(k, device=f'cuda:{self.local_rank}')
                            else:                                                   weights = torch.load(k, map_location=torch.device(f'cuda:{self.local_rank}'))
                            w = k.split("/")[-1]
                            a = w if not "." in w else w.split(".")[0]
                            names.append(a)
                            self.pipe.load_lora_weights(weights, weight_name=w, adapter_name=a, local_files_only=True, low_cpu_mem_usage=True)
                            self.log(f"✅ Added LoRA (scale={v}): {k}")

                        target.set_adapters(names, list(data["lora"].values()))
                        loaded_adapters = target.active_adapters()
                        loaded_adapters_string = ""
                        for la in loaded_adapters: loaded_adapters_string += "\n        " + la
                        self.log(f"📟 Total loaded LoRAs: {len(loaded_adapters)}")
                        self.log(f"📚 Adapters: {loaded_adapters_string}")
                        if len(names) > 0:  self.adapter_names = names
                        else:               self.adapter_names = None
                self.progress = 90

                # compiles & optimizations
                for k, v in self.pipe.components.items():
                    try: v.eval()
                    except: pass

                if self.pipeline_type in ["sd1", "sd2", "sdxl"] and data["sd_fuse_qkv_projections"] == True:
                    self.pipe.fuse_qkv_projections()
                    self.log("✅ Fused qkv projections")

                if data["compile_config"] is not None:
                    compile_transformer     = data["compile_config"].get("compile_transformer")
                    compile_vae             = data["compile_config"].get("compile_vae")
                    compile_encoder         = data["compile_config"].get("compile_encoder")
                    compile_backend         = data["compile_config"].get("compile_backend")
                    compile_mode            = data["compile_config"].get("compile_mode")
                    compile_options         = data["compile_config"].get("compile_options")
                    dynamic                 = data["compile_config"].get("dynamic")
                    fullgraph               = data["compile_config"].get("fullgraph")

                    if compile_options is not None and len(compile_options) > 5:
                        try:
                            compile_options = json.loads(compile_options)
                        except:
                            self.log("⚠️ Invalid JSON for compile_options - ignoring compile_options")
                            compile_options = None
                    else:
                        compile_options = None

                    if compile_mode is not None and compile_options is not None:
                        self.log("⚠️ compile_mode and compile_options are both defined - will ignore compile_mode")
                        compile_mode = None

                    compiler_config                             = {}
                    compiler_config["fullgraph"]                = fullgraph == True
                    compiler_config["dynamic"]                  = dynamic == True
                    if compile_backend is not None:             compiler_config["backend"] = compile_backend
                    if compile_mode is not None:                compiler_config["mode"] = compile_mode
                    if compile_options is not None:             compiler_config["options"] = compile_options

                    if compile_transformer:
                        if self.is_transformer_model_type == True:
                            self.compile_helper("transformer", compiler_config)
                        else:
                            self.compile_helper("unet", compiler_config)
                    if compile_vae:                             self.compile_helper("vae", compiler_config)
                    if compile_encoder:                         self.compile_helper("encoder", compiler_config)

                # clean up
                clean()

                # complete
                self.log("✅ Model initialization completed", rank_0_only=False)
                self.print_mem_usage()
                self.applied = data
                self.progress = 100

                return "", 200
            except:
                self.log(traceback.format_exc(), rank_0_only=False)
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

        self.log(f"ℹ️ Loading model: {model_path} with config: {str(config_path)}")
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
                if self.vae_dtype is not None:
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

        self.log(f"ℹ️ Loading model: {model_path} with config: {str(config_path)}")
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


    def compile_helper(self, target, compile_config):
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
                self.log("❌ Unknown compile target - not compiling")
                return
        self.log(f"⚙️ {target} will be compiled")
        return


    def print_params(self, data):
        formatted = "📋 Received parameters:"
        for k, v in data.items():
            sv = str(v)
            if v is None or len(sv) == 0:
                continue
            elif torch.is_tensor(v):
                formatted += f'\n        {k}: {str(v is not None)}'
            else:
                if sv.startswith("{") and sv.endswith("}"):
                    try:
                        sv = format_json(v, indent=8, indent_all=True)
                        formatted += f'\n        {k}: {sv}'
                    except:
                        formatted += f'\n        {k}: {str(v is not None)}'
                else:
                    if len(sv) < 128:
                        formatted += f'\n        {k}: {sv}'
                    else:
                        formatted += f'\n        {k}: {str(v is not None)}'
        self.log(formatted)
        return


    def prepare_inputs(self, data):
        data["height"]              = data.setdefault("height")
        data["width"]               = data.setdefault("width")
        data["positive"]            = data.setdefault("positive")
        data["negative"]            = data.setdefault("negative")
        data["positive_embeds"]     = data.setdefault("positive_embeds")
        data["negative_embeds"]     = data.setdefault("negative_embeds")
        data["image"]               = data.setdefault("image")
        data["ip_image"]            = data.setdefault("ip_image")
        data["control_image"]       = data.setdefault("control_image")
        data["latent"]              = data.setdefault("latent")
        data["steps"]               = data.setdefault("steps")
        data["cfg"]                 = data.setdefault("cfg")
        data["controlnet_scale"]    = data.setdefault("controlnet_scale")
        data["ip_adapter_scale"]    = data.setdefault("ip_adapter_scale")
        data["seed"]                = data.setdefault("seed")
        data["frames"]              = data.setdefault("frames")
        data["decode_chunk_size"]   = data.setdefault("decode_chunk_size")
        data["clip_skip"]           = data.setdefault("clip_skip")
        data["motion_bucket_id"]    = data.setdefault("motion_bucket_id")
        data["noise_aug_strength"]  = data.setdefault("noise_aug_strength")
        data["denoising_start"]     = data.setdefault("denoising_start")
        data["denoising_end"]       = data.setdefault("denoising_end")
        data["scheduler"]           = data.setdefault("scheduler")
        data["use_compel"]          = data.setdefault("use_compel")

        self.print_params(data)

        # checks
        assert not (data["image"] is None and data["positive"] is None and data["positive_embeds"] is None), "No input provided"
        assert not (data["positive"] is not None and data["positive_embeds"] is not None), "Provide only one positive input"
        assert not (data["negative"] is not None and data["negative_embeds"] is not None), "Provide only one negative input"
        assert not (data["image"] is None and self.pipeline_type in ["sdup", "svd", "wani2v"]), "No image provided for an image pipeline"
        assert not (data["ip_image"] is None and self.applied.get("ip_adapter") is not None), "No IPAdapter image provided for a IPAdapter-loaded pipeline"
        assert not (data["control_image"] is None and self.applied.get("control_net") is not None), "No ConstrolNet image provided for a ControlNet-loaded pipeline"

        data["image"]                                                                           = decode_b64_and_unpickle(data["image"])
        data["ip_image"]                                                                        = decode_b64_and_unpickle(data["ip_image"])
        data["control_image"]                                                                   = decode_b64_and_unpickle(data["control_image"])
        data["latent"]                                                                          = decode_b64_and_unpickle(data["latent"])
        data["positive_embeds"]                                                                 = decode_b64_and_unpickle(data["positive_embeds"])
        data["negative_embeds"]                                                                 = decode_b64_and_unpickle(data["negative_embeds"])
        if data["positive"] is not None and len(data["positive"]) == 0:                         data["positive"] = None
        if data["negative"] is not None and len(data["negative"]) == 0:                         data["negative"] = None
        if data["denoising_start"] is None or data["denoising_start"] < 0:                      data["denoising_start"] = 0
        if data["denoising_end"] is None or data["denoising_end"] > data["steps"]:              data["denoising_end"] = data["steps"]

        # load images
        if data["image"] is not None and self.pipeline_type in ["sdup", "svd", "wani2v"]:       data["image"] = load_image(data["image"])
        if data["ip_image"] is not None and self.applied.get("ip_adapter") is not None:         data["ip_image"] = load_image(data["ip_image"])
        if data["control_image"] is not None and self.applied.get("control_net") is not None:   data["control_image"] = load_image(data["control_image"])

        return data


    def print_mem_usage(self):
        mem_usage_string = f"🖥️ Memory usage:"
        for k, v in self.pipe.components.items():
            try:    mem = round(v.get_memory_footprint() / 1024 / 1024, 1)
            except: mem = "?"
            mem_usage_string += f"\n        {k}: {mem} MB"
        self.log(f"{mem_usage_string}")
        return


    def set_scheduler_timesteps(self, start=0):
        clean_override_function(self.pipe.__class__, 'retrieve_timesteps')
        if start is not None and start > 0:
            old_retrieve_timesteps = get_function_from_class(self.pipe.__class__, 'retrieve_timesteps')
            def new_retrieve_timesteps(*args, **kwargs):
                nonlocal start
                result = lambda: old_retrieve_timesteps(*args, **kwargs)
                timesteps, num_inference_steps = result()

                if len(timesteps) > 0 and start > 0:
                    timesteps = self.pipe.scheduler.timesteps[start * self.pipe.scheduler.order :]
                    if hasattr(self.pipe.scheduler, "set_begin_index"):
                        self.pipe.scheduler.set_begin_index(start * self.pipe.scheduler.order)
                return timesteps, num_inference_steps - start
            override_function(self.pipe.__class__, 'retrieve_timesteps', new_retrieve_timesteps)
        return


    def setup_inference(self, data, can_use_compel=True, callbacks={}):
        self.progress = 0

        # set scheduler
        if data["scheduler"] is not None:
            self.set_scheduler(data["scheduler"])
            self.log(f"ℹ️ Set scheduler to {get_scheduler_name(self.pipe.scheduler)}")
        elif self.pipe.scheduler != self.default_scheduler:
            self.pipe.scheduler = copy.deepcopy(self.default_scheduler)
            self.log(f"ℹ️ Reverted scheduler to {get_scheduler_name(self.pipe.scheduler)}")

        if data["denoising_start"] is not None and data["denoising_start"] > 0:
            if self.pipeline_type in DENOISING_START_WORKAROUND_PIPELINES:
                # workaround for pipelines that do not support denoising_start
                pass
            else:
                self.set_scheduler_timesteps(start=data["denoising_start"])
        else:
            self.set_scheduler_timesteps()

        # dirty patch zimage for heun
        if self.pipeline_type == "zimage" and data.get("scheduler") is not None:
            if json.loads(data["scheduler"])["scheduler"] == "fm_heun":
                # NOTE: copied from above - zimage uses latent inject when start>0, this will be fine...for now
                clean_override_function(self.pipe.__class__, 'retrieve_timesteps')
                old_retrieve_timesteps = get_function_from_class(self.pipe.__class__, 'retrieve_timesteps')
                def new_retrieve_timesteps(*args, **kwargs):
                    del kwargs["mu"]
                    result = lambda: old_retrieve_timesteps(*args, **kwargs)
                    timesteps, num_inference_steps = result()
                    return timesteps, num_inference_steps
                override_function(self.pipe.__class__, 'retrieve_timesteps', new_retrieve_timesteps)

        if data["latent"] is not None:
            data["latent"] = self.process_input_latent(data["latent"])

        # progress bar
        def set_step_progress(pipe, index, timestep, callback_kwargs):
            global DENOISING_START_WORKAROUND_PIPELINES
            nonlocal self, callbacks, data
            if torch.any(callback_kwargs["latents"].isnan()):
                self.log("⁉️ NaN detected in latents - stopping generation", rank_0_only=False)
                self.pipe._interrupt = True
                self.progress = 100
                return callback_kwargs
            the_index = index / self.pipe.scheduler.order

            # host callbacks
            for k, v in callbacks.items():
                if the_index == int(k):
                    try:    v(data, index, timestep, callback_kwargs)
                    except: pass

            # denoising_start workaround
            if data["latent"] is not None and self.pipeline_type in DENOISING_START_WORKAROUND_PIPELINES:
                if the_index == data["denoising_start"]:
                    self.log(f"ℹ️ Injecting latent at step {str(the_index)}", rank_0_only=False)
                    callback_kwargs["latents"] = data["latent"]
                elif the_index < data["denoising_start"]:
                    callback_kwargs["latents"] = torch.zeros_like(data["latent"])

            # denoising_end
            if data["denoising_end"] is not None and the_index >= data["denoising_end"]:
                self.log("ℹ️ Denoising end reached - stopping generation", rank_0_only=False)
                self.pipe._interrupt = True

            self.progress = int((the_index + data["denoising_start"]) / min(data["steps"], data["denoising_end"]) * 100)
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

        if data["positive_embeds"] is not None and data["negative_embeds"] is not None:
            dim_1 = max(data["positive_embeds"].size(1), data["negative_embeds"].size(1))
            pad_pos = max(0, dim_1 - data["positive_embeds"].size(1))
            pad_neg = max(0, dim_1 - data["negative_embeds"].size(1))
            if pad_pos > 0: data["positive_embeds"] = F.pad(data["positive_embeds"], (0, 0, 0, pad_pos))
            if pad_neg > 0: data["negative_embeds"] = F.pad(data["negative_embeds"], (0, 0, 0, pad_neg))

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
                    kwargs["negative_pooled_prompt_embeds"]     = data["negative_embeds"][0][1]["pooled_output"]
                    kwargs["negative_prompt_embeds"]            = data["negative_embeds"][0][0]
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
                    kwargs["negative_pooled_prompt_embeds"]            = data["negative_embeds"][0][1]["pooled_output"]
                    kwargs["negative_prompt_embeds"]                   = data["negative_embeds"][0][0]
            case _: # NOTE: "sd1", "sd2", "sd3", "sdxl"
                kwargs["output_type"]                           = "latent"
                kwargs["guidance_scale"]                        = data["cfg"]

                if self.pipeline_type in ["sd1", "sd2", "sd3", "sdxl"]:
                    kwargs["clip_skip"] = data["clip_skip"]

                if data["latent"] is not None:
                    kwargs["latents"] = data["latent"]
                else:
                    if data["height"] is not None:              kwargs["height"]                        = data["height"]
                    if data["width"] is not None:               kwargs["width"]                         = data["width"]

                if data["positive"] is not None:                kwargs["prompt"]                        = data["positive"]
                if data["negative"] is not None:                kwargs["negative_prompt"]               = data["negative"]
                if data["positive_embeds"] is not None:         kwargs["prompt_embeds"]                 = data["positive_embeds"]
                if data["positive_pooled_embeds"] is not None:  kwargs["pooled_prompt_embeds"]          = data["positive_pooled_embeds"]
                if data["negative_embeds"] is not None:         kwargs["negative_prompt_embeds"]        = data["negative_embeds"]
                if data["negative_pooled_embeds"] is not None:  kwargs["negative_pooled_prompt_embeds"] = data["negative_pooled_embeds"]
                if data["denoising_end"] is not None:           kwargs["denoising_end"]                 = float(data["denoising_end"] / data["steps"])

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
        # NOTE: this matters because it sometimes becomes NaN if you just move it to the target device
        latents = latents.to(dtype=torch.float16, device=torch.device("cpu"))
        if hasattr(self.pipe, "unet"):
            latents = latents.to(device=self.pipe.unet.device)
        else:
            latents = latents.to(device=self.pipe.transformer.device)
        latents = add_alpha_to_latent(latents)
        return latents


    def process_latent_for_output(self, latents, is_latent_output):
        latents = add_alpha_to_latent(latents)

        if is_latent_output:
            return latents

        if self.vae_dtype is not None:
            default_vae_dtype = self.pipe.vae.dtype
            self.pipe.vae = self.pipe.vae.to(dtype=self.vae_dtype)
        latents = latents.to(dtype=self.pipe.vae.dtype)
        latents = latents / self.pipe.vae.config.scaling_factor
        latents = self.pipe.vae.decode(latents, return_dict=False)[0]
        latents = self.pipe.image_processor.postprocess(latents, output_type="pil")
        if self.vae_dtype is not None:
            self.pipe.vae = self.pipe.vae.to(dtype=default_vae_dtype)
        return latents


    def convert_latent_to_output_latent(self, latents):
        latents = self.process_latent_for_output(latents, True)
        return latents


    def convert_latent_to_image(self, latents):
        latents = self.process_latent_for_output(latents, False)
        return latents
