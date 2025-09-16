import argparse
import base64
import copy
import gc
import os
import requests
import torch
import torch._dynamo
import torch.distributed as dist
from datetime import timedelta
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig, HunyuanDiT2DModel, QuantoConfig, SD3Transformer2DModel, Transformer2DModel
from diffusers.utils import load_image
from flask import Flask, request, jsonify
from optimum.quanto import freeze, quantize
from PIL import Image
from transformers import BertModel, CLIPTextModel, CLIPTextModelWithProjection, LlamaModel, T5EncoderModel


from xDiT.xfuser import xFuserArgs
from xDiT.xfuser.config import FlexibleArgumentParser
from xDiT.xfuser.core.distributed import init_distributed_environment, initialize_model_parallel


from Wan2_1 import wan
from Wan2_1.wan.configs import SIZE_CONFIGS
from Wan2_1.wan.configs.wan_i2v_14B import i2v_14B
from Wan2_1.wan.configs.wan_t2v_1_3B import t2v_1_3B
from Wan2_1.wan.configs.wan_t2v_14B import t2v_14B
from Wan2_1.wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from Wan2_1.wan.utils.utils import cache_image, cache_video


from modules.host_generics import *
from modules.scheduler_config import *
from modules.utils import *


app = Flask(__name__)
engine_config = None
initialized = False
step_progress = 0
input_config = None
local_rank = None
logger = None
pipe = None
result = None


is_14B_model = False
wan_model_type = ""
RESOLUTIONS = {
    "720x1280": (720, 1280),
    "1280x720": (1280, 720),
    "480x832": (480, 832),
    "832x480": (832, 480),
}


def get_args():
    parser = argparse.ArgumentParser()
    # xdit-usp
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    #generic
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
        global pipe, engine_config, input_config, local_rank, initialized, cache_args, logger, is_14B_model, wan_model_type, RESOLUTIONS
        args = get_args()
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        logger = get_logger(local_rank)

        # checks
        assert args.ulysses_degree * args.ring_degree == world_size, f"ulysses_degree and ring_degree should be equal to the world size."

        # set torch type
        torch_dtype = get_torch_type(args.variant)

        # init distributed inference
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size, timeout=timedelta(days=1))
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=dist.get_world_size(), ring_degree=args.ring_degree, ulysses_degree=args.ulysses_degree)

        if dist.is_initialized():
            base_seed = [1] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)

        # dynamo tweaks
        setup_torch_dynamo()
        # torch tweaks
        setup_torch_backends()

        # quantize
        # TODO

        # set pipeline
        logger.info(f"Initializing pipeline")
        # https://github.com/Wan-Video/Wan2.1
        kwargs = {}
        kwargs["checkpoint_dir"] = args.checkpoint
        kwargs["device_id"] = local_rank
        kwargs["rank"] = rank
        kwargs["t5_fsdp"] = False # True default
        kwargs["dit_fsdp"] = False # True default
        kwargs["use_usp"] = True # True default, main parallelization strategy should always be True
        kwargs["t5_cpu"] = True # False default, will not combine with t5_fsdp
        PipelineClass = None
        frame_num = 81
        match args.type:
            case "wan_t2v":
                is_14B_model = True
                wan_model_type = "text"
                PipelineClass = wan.WanT2V
                cfg = t2v_14B # TODO: support 1.3B
            case "wan_t2i":
                is_14B_model = True
                wan_model_type = "text"
                frame_num = 1
                PipelineClass = wan.WanT2V
                cfg = t2v_14B # TODO: support 1.3B
            case "wan_i2v":
                is_14B_model = True
                wan_model_type = "image"
                PipelineClass = wan.WanI2V
                cfg = i2v_14B
            case "wan_flf2v":
                is_14B_model = True
                wan_model_type = "image"
                PipelineClass = wan.WanFLF2V
                cfg = i2v_14B
            case "wan_vace":
                is_14B_model = True
                wan_model_type = "vace"
                PipelineClass = wan.WanVace
                cfg = t2v_14B # TODO: support 1.3B
            case "hyv":
                return
            case _:
                return

        assert cfg.num_heads % args.ulysses_degree == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_degree=}`."
        if not is_14B_model:
            assert args.width < 720 or args.height < 720, "720p video is not supported in 1.3B"

        if args.gguf_model is not None:
            kwargs["gguf"] = args.gguf_model
        cfg.param_dtype = torch_dtype
        if wan_model_type == "image":
            cfg.clip_dtype = torch_dtype
        if kwargs["t5_cpu"] == True:    cfg.t5_dtype = torch.float32
        else:                           cfg.t5_dtype = torch_dtype
        kwargs["config"] = cfg
        pipe = PipelineClass(**kwargs)
        logger.info(f"Pipeline initialized")

        # set memory saving
        # TODO

        # set ipadapter
        # TODO

        # set scheduler
        # TODO

        # set vae
        # TODO

        # set lora
        # TODO

        # compiles
        if args.compile_unet:           compile_model(pipe, logger)
        if args.compile_vae:            logger.info("vae compile not supported - ignoring") # compile_vae(pipe, logger)
        if args.compile_text_encoder:   compile_text_encoder(pipe, logger)

        # set progress bar visibility
        #pipe.set_progress_bar_config(disable=local_rank != 0)

        # clean up
        clean()

        # warm up run
        if args.warm_up_steps > 0:
            logger.info("Starting warmup run")
            warm_up_kwargs = {}
            warm_up_kwargs["input_prompt"] = "a dog"
            warm_up_kwargs["size"] = RESOLUTIONS[f"{args.height}x{args.width}"]
            warm_up_kwargs["frame_num"] = frame_num
            warm_up_kwargs["shift"] = 5.0
            warm_up_kwargs["sample_solver"] = "dpm++"
            warm_up_kwargs["sampling_steps"] = args.warm_up_steps
            warm_up_kwargs["guide_scale"] = 7.0
            warm_up_kwargs["seed"] = 1
            warm_up_kwargs["offload_model"] = True
            pipe.generate(**warm_up_kwargs)

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
    image,
    image_scale,
    steps,
    seed,
    cfg,
    frames
):
    with torch.no_grad():
        # TODO: use image
        global pipe, local_rank, input_config, result, step_progress
        args = get_args()
        torch.cuda.reset_peak_memory_stats()
        step_progress = 0

        def set_step_progress(pipe, index, timestep, callback_kwargs):
            global step_progress
            step_progress = index / steps * 100
            return callback_kwargs

        generator = torch.Generator(device="cpu").manual_seed(seed)

        kwargs                          = {}
        kwargs["generator"]             = generator
        kwargs["guidance_scale"]        = cfg
        kwargs["num_inference_steps"]   = steps
        kwargs["callback_on_step_end"]  = set_step_progress
        kwargs["width"]                 = args.width
        kwargs["height"]                = args.height
        if positive is not None:        kwargs["prompt"]            = positive
        if negative is not None:        kwargs["negative_prompt"]   = negative
        match args.type:
            case "wan_t2v":
                video = wan_t2v.generate(
                    args.prompt,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model)

            case "wan_i2v":
                # generator
                video = wan_i2v.generate(
                    args.prompt,
                    img,
                    max_area=args.height*args.width,
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model)

            case "wan_flf2v":
                # generator
                video = wan_flf2v.generate(
                    args.prompt,
                    first_frame,
                    last_frame,
                    max_area=args.height*args.width,
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model)

            case "wan_vace":
                # generator
                src_video, src_mask, src_ref_images = wan_vace.prepare_source(
                    [args.src_video], [args.src_mask], [
                        None if args.src_ref_images is None else
                        args.src_ref_images.split(',')
                    ], args.frame_num, SIZE_CONFIGS[args.size], local_rank)

                logging.info(f"Generating video...")
                video = wan_vace.generate(
                    args.prompt,
                    src_video,
                    src_mask,
                    src_ref_images,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model)


        if rank == 0:
            if args.save_file is None:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                        "_")[:50]
                suffix = '.png' if "t2i" in args.task else '.mp4'
                args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

            if "t2i" in args.task:
                logging.info(f"Saving generated image to {args.save_file}")
                cache_image(
                    tensor=video.squeeze(1)[None],
                    save_file=args.save_file,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
            else:
                logging.info(f"Saving generated video to {args.save_file}")
                cache_video(
                    tensor=video[None],
                    save_file=args.save_file,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
        logging.info("Finished.")

        clean()

        if local_rank == 0:
            while True:
                if result:
                    step_progress = 100
                    output_base64 = result
                    result = None
                    return "OK", output_base64, is_image
        elif output is not None:
            logger.info(f"task completed")
            if is_image:    o = output.images[0]
            else:           o = output.frames[0]
            output_base64 = pickle_and_encode_b64(o)
            with app.app_context():
                requests.post(f"http://localhost:{args.port}/set_result", json={"output": output_base64})


@app.route("/set_result", methods=["POST"])
def set_result():
    global result
    data = request.json
    result = data.get("output")
    return "", 200


@app.route("/generate", methods=["POST"])
def generate_image():
    global logger
    data = request.json
    dummy               = 0
    positive            = data.get("positive")
    negative            = data.get("negative")
    image               = data.get("image")
    image_scale         = data.get("image_scale")
    steps               = data.get("steps")
    seed                = data.get("seed")
    cfg                 = data.get("cfg")
    frames              = data.get("frames")

    if positive is not None and len(positive) == 0: positive = None
    if negative is not None and len(negative) == 0: negative = None
    if image is None and positive is None:          return jsonify({ "message": "No input provided", "output": None, "is_image": False })
    if image is not None:                           image = decode_b64_and_unpickle(image)

    params = [
        dummy,
        positive,
        negative,
        image,
        image_scale,
        steps,
        seed,
        cfg,
        frames
    ]
    dist.broadcast_object_list(params, src=0)
    print_params(
        {
            "positive": positive,
            "negative": negative,
            "image": (image is not None),
            "image_scale": image_scale,
            "steps": steps,
            "seed": seed,
            "cfg": cfg,
            "frames": frames,
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
            params = [None] * 9 # len(params) of generate_image_parallel()
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
