import argparse
import base64
import os
import pickle
import torch
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp
from compel import Compel, ReturnedEmbeddingsType
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


def get_args():
    parser = argparse.ArgumentParser()
    # distrifuser
    parser.add_argument("--no_cuda_graph", action="store_true")
    parser.add_argument("--no_split_batch", action="store_true")
    parser.add_argument("--parallelism", type=str, default="patch", choices=["patch", "tensor", "naive_patch"])
    parser.add_argument("--split_scheme", type=str, default="row", choices=["row", "col", "alternate"])
    parser.add_argument("--sync_mode", type=str, default="corrected_async_gn", choices=["separate_gn", "stale_gn", "corrected_async_gn", "sync_gn", "full_sync", "no_sync"])
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
            global pipe, local_rank, initialized, logger
            args = get_args()

            # checks
            # TODO: checks

            # init distributed inference
            mp.set_start_method("spawn", force=True)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            logger = get_logger(local_rank)

            # dynamo tweaks
            setup_torch_dynamo(args.torch_cache_limit, args.torch_accumlated_cache_limit, args.torch_capture_scalar)
            # torch tweaks
            setup_torch_backends()

            # set torch type
            torch_dtype = get_torch_type(args.variant)

            # set distrifuser
            warmup_steps = 0 if args.warm_up_steps is None else args.warm_up_steps
            distri_config = DistriConfig(
                height=args.height,
                width=args.width,
                do_classifier_free_guidance=True,
                split_batch=not args.no_split_batch,
                warmup_steps=warmup_steps,
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

            kwargs_vae = {}
            kwargs_vae["torch_dtype"] = torch.float32
            kwargs_vae["use_safetensors"] = True
            kwargs_vae["local_files_only"] = True
            kwargs_vae["low_cpu_mem_usage"] = True

            match args.type:
                case "sdxl":
                    kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_vae)
                    PipelineClass = DistriSDXLPipeline
                case _:
                    kwargs["vae"] = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae", **kwargs_vae)
                    PipelineClass = DistriSDPipeline
            kwargs["unet"] = UNet2DConditionModel.from_pretrained(args.checkpoint, subfolder="unet", **kwargs_model)

            # set vae
            if args.vae is not None:
                kwargs["vae"] = AutoencoderKL.from_pretrained(args.vae, **kwargs_vae)

            # init pipe
            pipe = PipelineClass.from_pretrained(distri_config=distri_config, **kwargs)
            logger.info(f"Pipeline initialized")

            # set scheduler
            set_scheduler(args, pipe, is_distrifuser=True)

            # set ipadapter
            if args.ip_adapter is not None:
                args.ip_adapter = json.loads(args.ip_adapter)
                load_ip_adapter(pipe, args.ip_adapter)

            # set memory saving
            if args.enable_vae_slicing:             pipe.pipeline.vae.enable_slicing()
            if args.enable_vae_tiling:              pipe.pipeline.vae.enable_tiling()
            if args.xformers_efficient:             pipe.pipeline.enable_xformers_memory_efficient_attention()
            if args.enable_sequential_cpu_offload:  logger.info("sequential CPU offload not supported - ignoring")
            if args.enable_model_cpu_offload:       logger.info("model CPU offload not supported - ignoring")

            # quantize
            if args.quantize_unet_to is not None:       quantize_helper("unet", pipe, args.quantize_unet_to, logger, is_distrifuser=True)
            if args.quantize_encoder_to is not None:    quantize_helper("encoder", pipe, args.quantize_encoder_to, logger)
            if args.quantize_misc_to is not None:       pass

            # set lora
            adapter_names = None
            if args.lora is not None:
                adapter_names = load_lora(args.lora, pipe.pipeline, local_rank, logger)

            # compiles
            if args.compile_unet or args.compile_vae or args.compile_encoder:
                if args.compile_mode is not None and args.compile_options is not None:
                    logger.info("Compile mode and options are both defined, will ignore compile mode.")
                    args.compile_mode = None
                compiler_config = {}
                if args.compile_backend is not None:    compiler_config["backend"] = args.compile_backend
                if args.compile_mode is not None:       compiler_config["mode"] = args.compile_mode
                if args.compile_options is not None:    compiler_config["options"] = json.loads(args.compile_options)
                compiler_config["fullgraph"] = (args.compile_fullgraph_off is None or args.compile_fullgraph_off == False)
                compiler_config["dynamic"] = False
                if args.compile_unet:                   compile_helper("unet", pipe, compiler_config, logger, adapter_names=adapter_names)
                if args.compile_vae:                    compile_helper("vae", pipe, compiler_config, logger)
                if args.compile_encoder:                compile_helper("encoder", pipe, compiler_config, logger)

            # set progress bar visibility
            pipe.set_progress_bar_config(disable=distri_config.rank != 0)

            # clean up
            clean()

            # warm up run
            if args.warm_up_steps is not None and args.warm_up_steps > 0:
                logger.info("Starting warmup run")
                kwargs = {}
                kwargs["prompt"] = "a dog"
                kwargs["num_inference_steps"] = args.warm_up_steps
                kwargs["guidance_scale"] = 7
                kwargs["generator"] = torch.Generator(device="cpu").manual_seed(1)
                kwargs["output_type"] = "pil"
                if args.ip_adapter is not None:
                    kwargs["ip_adapter_image"] = get_warmup_image()
                pipe.pipeline.vae = pipe.pipeline.vae.to(torch_dtype)
                pipe(**kwargs)
                pipe.pipeline.vae = pipe.pipeline.vae.to(torch.float32)

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
    sigmas,
    timesteps,
    denoising_start,
    denoising_end,
):
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            global pipe, local_rank, step_progress
            args = get_args()
            device = torch.device("cuda", torch.cuda.current_device())
            torch_dtype = get_torch_type(args.variant)
            torch.cuda.reset_peak_memory_stats()
            step_progress = 0
            set_scheduler(args, pipe, is_distrifuser=True)

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

            if args.compel and positive_embeds is None and negative_embeds is None:
                if args.type in ["sd1", "sd2"]: embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
                else:                           embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
                compel = Compel(
                    tokenizer=[pipe.pipeline.tokenizer, pipe.pipeline.tokenizer_2],
                    text_encoder=[pipe.pipeline.text_encoder, pipe.pipeline.text_encoder_2],
                    returned_embeddings_type=embeddings_type,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False,
                )
                positive_embeds, positive_pooled_embeds = compel([positive])
                if negative is not None and len(negative) > 0:  negative_embeds, negative_pooled_embeds = compel([negative])
                positive = negative = None

            def set_step_progress(pipe, index, timestep, callback_kwargs):
                global get_torch_type, logger, process_input_latent, step_progress
                nonlocal args, denoising_start, device, latent, steps, torch_dtype
                scheduler_name = get_scheduler_name(pipe.scheduler)
                the_index = get_scheduler_progressbar_offset_index(pipe.scheduler, index)
                step_progress = the_index / steps * 100
                if latent is not None:
                    if denoising_start is None or denoising_start > 1.0:
                        denoising_start = 1.0
                    target = int(steps * (1 - denoising_start))
                    if index == target:
                        latent = process_input_latent(latent, pipe, torch_dtype, device, timestep=timestep, is_distrifuser=True)
                        callback_kwargs["latents"] = latent
                        logger.info(f'Injected latent at step {target}')
                return callback_kwargs

            generator = torch.Generator(device="cpu").manual_seed(seed)

            kwargs                                          = {}
            kwargs["generator"]                             = generator
            kwargs["guidance_scale"]                        = cfg
            kwargs["num_inference_steps"]                   = steps
            kwargs["callback_on_step_end"]                  = set_step_progress
            kwargs["callback_on_step_end_tensor_inputs"]    = ["latents"]
            kwargs["output_type"]                           = "latent"
            if positive is not None:                        kwargs["prompt"]                    = positive
            if negative is not None:                        kwargs["negative_prompt"]           = negative
            if positive_embeds is not None:                 kwargs["prompt_embeds"]             = positive_embeds
            if positive_pooled_embeds is not None:          kwargs["pooled_prompt_embeds"]      = positive_pooled_embeds
            if negative_embeds is not None:                 kwargs["negative_embeds"]           = negative_embeds
            if negative_pooled_embeds is not None:          kwargs["negative_pooled_embeds"]    = negative_pooled_embeds
            if clip_skip is not None:                       kwargs["clip_skip"]                 = clip_skip
            if sigmas is not None:                          kwargs["sigmas"]                    = sigmas
            if timesteps is not None:                       kwargs["timesteps"]                 = timesteps
            if denoising_end is not None:                   kwargs["denoising_end"]             = denoising_end
            if args.ip_adapter is not None:
                if ip_image is not None:    kwargs["ip_adapter_image"] = ip_image
                else:                       return "No IPAdapter image provided for a IPAdapter-loaded pipeline", None, False

            output = pipe(**kwargs)

            if args.compel:
                # https://github.com/damian0815/compel/issues/24
                del positive_embeds, positive_pooled_embeds, negative_embeds, negative_pooled_embeds, compel

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
                    images = convert_latent_to_image(copy.copy(output.images.to(device)), pipe, is_distrifuser=True)
                    latents = convert_latent_to_output_latent(copy.copy(output.images.to(device)), pipe, is_distrifuser=True)
                    return "OK", { "image": pickle_and_encode_b64(images[0]), "latent": pickle_and_encode_b64(latents) }, True
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
    sigmas              = data.get("sigmas")
    timesteps           = data.get("timesteps")
    denoising_start     = data.get("denoising_start")
    denoising_end       = data.get("denoising_end")

    print_params(data, logger)

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
        sigmas,
        timesteps,
        denoising_start,
        denoising_end,
    ]
    dist.broadcast_object_list(params, src=0)
    message, outputs, is_image = generate_image_parallel(*params)
    response = { "message": message, "output": outputs.get("image"), "latent": outputs.get("latent"), "is_image": is_image }
    return jsonify(response)


def run_host():
    global logger
    args = get_args()
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        app.run(host="localhost", port=args.port)
    else:
        while True:
            params = [None] * 15 # len(params) of generate_image_parallel()
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
