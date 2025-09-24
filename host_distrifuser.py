import argparse
import base64
import os
import pickle
import torch
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp
from compel import Compel, ReturnedEmbeddingsType
from DeepCache import DeepCacheSDHelper
from diffusers import (
    AutoencoderKL,
    QuantoConfig,
    UNet2DConditionModel,
)
from diffusers.utils import load_image
from flask import Flask, request, jsonify
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection


from DistriFuser.distrifuser.utils import DistriConfig
from DistriFuser.distrifuser.pipelines import DistriSDPipeline, DistriSDXLPipeline


from modules.host_generics import *
from modules.scheduler_config import *
from modules.utils import *


app = Flask(__name__)
initialized = False
step_progress = 0
local_rank = None
logger = None
pipe = None
can_cache = False # TODO: fix pipe has no attr 'unet'
cache_kwargs = { "cache_interval": 3, "cache_branch_id": 0 }


def get_args():
    parser = argparse.ArgumentParser()
    # distrifuser
    parser.add_argument("--no_cuda_graph", action="store_true")
    parser.add_argument("--no_split_batch", action="store_true")
    parser.add_argument("--parallelism", type=str, default="patch", choices=["patch", "tensor", "naive_patch"])
    parser.add_argument("--split_scheme", type=str, default="row", choices=["row", "col", "alternate"])
    parser.add_argument("--sync_mode", type=str, default="corrected_async_gn", choices=[
                                                                                            "separate_gn",
                                                                                            "stale_gn",
                                                                                            "corrected_async_gn",
                                                                                            "sync_gn",
                                                                                            "full_sync",
                                                                                            "no_sync"
                                                                                        ])
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
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            global pipe, local_rank, initialized, can_cache, cache_kwargs, logger
            args = get_args()

            # checks
            # TODO: checks

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
            if args.quantize_to is not None:    quant = {"quantization_config": QuantoConfig(weights_dtype=args.quantize_to)}
            def quantize(model, desc):          do_quantization(model, desc, args.quantize_to, logger)

            # set distrifuser
            distri_config = DistriConfig(
                height=args.height,
                width=args.width,
                do_classifier_free_guidance=True,
                split_batch=not args.no_split_batch,
                warmup_steps=args.warm_up_steps,
                comm_checkpoint=60,
                mode=args.sync_mode,
                use_cuda_graph=not args.no_cuda_graph,
                parallelism=args.parallelism,
                split_scheme=args.split_scheme,
                verbose=True,
            )

            # set pipeline
            logger.info(f"Initializing pipeline")
            kwargs = {}
            kwargs["pretrained_model_name_or_path"] = args.checkpoint
            kwargs["torch_dtype"] = torch_dtype
            kwargs["use_safetensors"] = True
            kwargs["local_files_only"] = True
            kwargs["low_cpu_mem_usage"] = True

            kwargs_model = {}
            kwargs_model["torch_dtype"] = torch_dtype
            kwargs_model["use_safetensors"] = True
            kwargs_model["local_files_only"] = True
            kwargs_model["low_cpu_mem_usage"] = True

            to_quantize = {}
            quantize_unet_after = False

            match args.type:
                case "sdxl":
                    if args.quantize_to is not None:
                        to_quantize["text_encoder"] = CLIPTextModel.from_pretrained(args.checkpoint, subfolder="text_encoder", **kwargs_model)
                        to_quantize["text_encoder_2"] = CLIPTextModelWithProjection.from_pretrained(args.checkpoint, subfolder="text_encoder_2", **kwargs_model)
                        if args.ip_adapter is None:
                            to_quantize["unet"] = UNet2DConditionModel.from_pretrained(args.checkpoint, subfolder="unet", **quant, **kwargs_model).to(f'cuda:{local_rank}')
                        else:
                            quantize_unet_after = True
                    kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_model)
                    PipelineClass = DistriSDXLPipeline
                case _:
                    if args.quantize_to is not None:
                        to_quantize["text_encoder"] = CLIPTextModel.from_pretrained(args.checkpoint, subfolder="text_encoder", **kwargs_model)
                        if args.ip_adapter is None:
                            to_quantize["unet"] = UNet2DConditionModel.from_pretrained(args.checkpoint, subfolder="unet", **quant, **kwargs_model).to(f'cuda:{local_rank}')
                        else:
                            quantize_unet_after = True
                    kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_model)
                    PipelineClass = DistriSDPipeline

            # set vae
            if args.vae is not None:
                kwargs["vae"] = AutoencoderKL.from_pretrained(args.vae, **kwargs_model)

            if len(to_quantize) > 0:
                for k, v in to_quantize.items():
                    quantize(v, k)
                    kwargs[k] = v

            pipe = PipelineClass.from_pretrained(distri_config=distri_config, **quant, **kwargs)
            logger.info(f"Pipeline initialized")

            # set ipadapter
            if args.ip_adapter is not None:
                args.ip_adapter = json.loads(args.ip_adapter)
                load_ip_adapter(pipe, args.ip_adapter)

            # deferred quantize
            if quantize_unet_after:
                quantize(pipe.unet, "unet")

            # set lora
            adapter_names = None
            if args.lora is not None:
                adapter_names = load_lora(args.lora, pipe.pipeline, local_rank, logger, (args.quantize_to is not None))
                if len(to_quantize) > 0:
                    logger.info("Requantizing unet, text encoder(s)")
                    if to_quantize.get("unet") is not None:             quantize(pipe.unet, "unet")
                    if to_quantize.get("text_encoder") is not None:     quantize(pipe.text_encoder, "text_encoder")
                    if to_quantize.get("text_encoder_2") is not None:   quantize(pipe.text_encoder_2, "text_encoder_2")

            # set scheduler
            if args.scheduler is not None:
                args.scheduler = json.loads(args.scheduler)
                pipe.pipeline.scheduler = get_scheduler(args.scheduler, pipe.pipeline.scheduler.config)

            # set memory saving
            if args.enable_vae_slicing:             pipe.pipeline.vae.enable_slicing()
            if args.enable_vae_tiling:              pipe.pipeline.vae.enable_tiling()
            if args.xformers_efficient:             pipe.pipeline.enable_xformers_memory_efficient_attention()
            if args.enable_sequential_cpu_offload:  logger.info("sequential CPU offload not supported - ignoring")
            if args.enable_model_cpu_offload:       logger.info("model CPU offload not supported - ignoring")

            # compiles
            if args.compile_unet:           compile_unet(pipe.pipeline, adapter_names, logger, is_distrifuser=True)
            if args.compile_vae:            compile_vae(pipe.pipeline, logger)
            if args.compile_text_encoder:   compile_text_encoder(pipe.pipeline, logger)

            # set progress bar visibility
            pipe.set_progress_bar_config(disable=distri_config.rank != 0)

            # clean up
            clean()

            # warm up run
            if args.warm_up_steps is not None and args.warm_up_steps > 0:
                logger.info("Starting warmup run")
                if can_cache:
                    helper = DeepCacheSDHelper(pipe=pipe)
                    helper.set_params(**cache_kwargs)
                    helper.enable()
                kwargs = {}
                kwargs["prompt"] = "a dog"
                kwargs["num_inference_steps"] = args.warm_up_steps
                kwargs["guidance_scale"] = 7
                kwargs["generator"] = torch.Generator(device="cpu").manual_seed(1)
                kwargs["output_type"] = "pil"
                if args.ip_adapter is not None:
                    kwargs["ip_adapter_image"] = get_warmup_image()
                pipe(**kwargs)
                if can_cache:
                    helper.disable()

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
    ip_image,
    latent,
    steps,
    seed,
    cfg,
    clip_skip,
    denoise,
    sigmas,
    timesteps,
):
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            global pipe, local_rank, step_progress, can_cache, cache_kwargs
            args = get_args()
            torch.cuda.reset_peak_memory_stats()
            step_progress = 0

            if ip_image is not None and args.ip_adapter is not None:
                ip_image = load_image(ip_image)

            positive_pooled_embeds = None
            negative_pooled_embeds = None
            if positive_embeds is not None:
                positive_pooled_embeds = positive_embeds[0][1]["pooled_output"]
                positive_embeds = positive_embeds[0][0]
            if negative_embeds is not None:
                negative_pooled_embeds = negative_embeds[0][1]["pooled_output"]
                negative_embeds = negative_embeds[0][0]

            if args.compel and not positive_embeds and not negative_embeds:
                if args.type in ["sd1", "sd2"]:
                    embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
                else:
                    embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
                compel = Compel(
                    tokenizer=[pipe.pipeline.tokenizer, pipe.pipeline.tokenizer_2],
                    text_encoder=[pipe.pipeline.text_encoder, pipe.pipeline.text_encoder_2],
                    returned_embeddings_type=embeddings_type,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False,
                )
                positive_embeds, positive_pooled_embeds = compel([positive])
                if negative is not None and len(negative) > 0:
                    negative_embeds, negative_pooled_embeds = compel([negative])
                positive = None
                negative = None

            def set_step_progress(pipe, index, timestep, callback_kwargs):
                global step_progress
                step_progress = index / steps * 100
                return callback_kwargs

            generator = torch.Generator(device="cpu").manual_seed(seed)

            kwargs                                  = {}
            kwargs["generator"]                     = generator
            kwargs["guidance_scale"]                = cfg
            kwargs["num_inference_steps"]           = steps
            kwargs["callback_on_step_end"]          = set_step_progress
            if positive is not None:                kwargs["prompt"]                    = positive
            if negative is not None:                kwargs["negative_prompt"]           = negative
            if positive_embeds is not None:         kwargs["prompt_embeds"]             = positive_embeds
            if positive_pooled_embeds is not None:  kwargs["pooled_prompt_embeds"]      = positive_pooled_embeds
            if negative_embeds is not None:         kwargs["negative_embeds"]           = negative_embeds
            if negative_pooled_embeds is not None:  kwargs["negative_pooled_embeds"]    = negative_pooled_embeds
            if clip_skip is not None:               kwargs["clip_skip"]                 = clip_skip
            if sigmas is not None:                  kwargs["sigmas"]                    = sigmas
            if timesteps is not None:               kwargs["timesteps"]                 = timesteps
            if latent is not None:
                latent = process_input_latent(
                    latent,
                    get_scheduler_name(pipe.pipeline.scheduler),
                    pipe,
                    get_torch_type(args.variant),
                    torch.device("cuda", torch.cuda.current_device()),
                    is_distrifuser=True,
                )
                kwargs["latents"] = latent
            if denoise is not None:
                latent, result = set_timesteps(
                    pipe,
                    latent,
                    get_scheduler_name(pipe.scheduler),
                    steps,
                    denoise,
                    sigmas,
                    timesteps,
                    logger,
                    is_distrifuser=True
                )
                if latent is not None: kwargs["latents"] = latent
                if result is not None: kwargs.update(result)
            if args.ip_adapter is not None:
                if ip_image is not None:
                    kwargs["ip_adapter_image"] = ip_image
                else:
                    return "No IPAdapter image provided for a IPAdapter-loaded pipeline", None, False

            if can_cache:
                helper = DeepCacheSDHelper(pipe=pipe)
                helper.set_params(**cache_kwargs)
                helper.enable()
            output = pipe(**kwargs)
            if can_cache:
                helper.disable()

            if args.compel:
                # https://github.com/damian0815/compel/issues/24
                positive_embeds = positive_pooled_embeds = negative_embeds = negative_pooled_embeds = None

            # clean up
            clean()

            if dist.get_rank() != 0:
                # serialize output object
                output_bytes = pickle.dumps(output)

                # send output to rank 0
                dist.send(torch.tensor(len(output_bytes), device=f"cuda:{local_rank}"), dst=0)
                dist.send(torch.ByteTensor(list(output_bytes)).to(f"cuda:{local_rank}"), dst=0)

                logger.info("Output sent to rank 0")

            elif dist.get_rank() == 0 and dist.get_world_size() > 1:
                # recv from rank world_size - 1
                size = torch.tensor(0, device=f"cuda:{local_rank}")
                dist.recv(size, src=dist.get_world_size() - 1)
                output_bytes = torch.ByteTensor(size.item()).to(f"cuda:{local_rank}")
                dist.recv(output_bytes, src=dist.get_world_size() - 1)

                # deserialize output object
                output = pickle.loads(output_bytes.cpu().numpy().tobytes())
                if output is not None:
                    step_progress = 100
                    return "OK", pickle_and_encode_b64(output.images[0]), True
                else:
                    return "No image from pipeline", None, True


@app.route("/generate", methods=["POST"])
def generate_image():
    global logger
    data = request.json
    dummy               = 0
    positive            = data.get("positive")
    negative            = data.get("negative")
    positive_embeds     = data.get("positive_embeds")
    negative_embeds     = data.get("negative_embeds")
    ip_image            = data.get("ip_image")
    latent              = data.get("latent")
    steps               = data.get("steps")
    seed                = data.get("seed")
    cfg                 = data.get("cfg")
    clip_skip           = data.get("clip_skip")
    denoise             = data.get("denoise")
    sigmas              = data.get("sigmas")
    timesteps           = data.get("timesteps")

    if positive is not None and len(positive) == 0:             positive = None
    if negative is not None and len(negative) == 0:             negative = None
    if positive is None and positive_embeds is None:            jsonify({ "message": "No input provided", "output": None, "is_image": False })
    if positive is not None and positive_embeds is not None:    jsonify({ "message": "Provide only one positive input", "output": None, "is_image": False })
    if negative is not None and negative_embeds is not None:    jsonify({ "message": "Provide only one negative input", "output": None, "is_image": False })
    if ip_image is not None:                                    ip_image = decode_b64_and_unpickle(ip_image)
    if latent is not None:                                      latent = decode_b64_and_unpickle(latent)
    if positive_embeds is not None:                             positive_embeds = decode_b64_and_unpickle(positive_embeds)
    if negative_embeds is not None:                             negative_embeds = decode_b64_and_unpickle(negative_embeds)
    if sigmas is not None:                                      sigmas = decode_b64_and_unpickle(sigmas)
    if timesteps is not None:                                   timesteps = decode_b64_and_unpickle(timesteps)

    params = [
        dummy,
        positive,
        negative,
        positive_embeds,
        negative_embeds,
        ip_image,
        latent,
        steps,
        seed,
        cfg,
        clip_skip,
        denoise,
        sigmas,
        timesteps,
    ]
    dist.broadcast_object_list(params, src=0)
    print_params(
        {
            "positive": positive,
            "negative": negative,
            "positive_embeds": (positive_embeds is not None),
            "negative_embeds": (negative_embeds is not None),
            "ip_image": (ip_image is not None),
            "latent": (latent is not None),
            "steps": steps,
            "seed": seed,
            "cfg": cfg,
            "clip_skip": clip_skip,
            "denoise": denoise,
            "sigmas": (sigmas is not None),
            "timesteps": (timesteps is not None),
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
            params = [None] * 14 # len(params) of generate_image_parallel()
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
