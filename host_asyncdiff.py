import argparse
import base64
import copy
import json
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback
from datetime import timedelta
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
from diffusers.utils import load_image
from flask import Flask, request, jsonify
from PIL import Image


from AsyncDiff.asyncdiff.async_animate import AsyncDiff as AsyncDiffAnimateDiff
from AsyncDiff.asyncdiff.async_flux import AsyncDiff as AsyncDiffFlux
from AsyncDiff.asyncdiff.async_sd import AsyncDiff as AsyncDiffStableDiffusion
from AsyncDiff.asyncdiff.async_sd3 import AsyncDiff as AsyncDiffStableDiffusion3
from AsyncDiff.asyncdiff.async_wan import AsyncDiff as AsyncDiffWan
from AsyncDiff.asyncdiff.async_zimage import AsyncDiff as AsyncDiffZImage


from modules.host_common import *
from modules.scheduler_config import *
from modules.utils import *


app = Flask(__name__)
async_diff = None
base = None


def __initialize_distributed_environment():
    global base
    base = CommonHost()
    mp.set_start_method("spawn", force=True)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", timeout=timedelta(days=1))
    base.local_rank = dist.get_rank()
    base.set_logger()
    base.initialized = True
    return


args = None
def __run_host():
    global args, base

    parser = argparse.ArgumentParser()
    # asyncdiff
    parser.add_argument("--model_n",        type=int,   default=2) # NOTE: if n > 4, you'll need to manually map your model in pipe_config.py
    parser.add_argument("--stride",         type=int,   default=1)
    parser.add_argument("--synced_steps",   type=int,   default=3)
    parser.add_argument("--synced_percent", type=float, default=None)
    parser.add_argument("--time_shift",     action="store_true")
    # generic
    for k, v in GENERIC_HOST_ARGS.items():  parser.add_argument(f"--{k}", type=v, default=None)
    for e in GENERIC_HOST_ARGS_TOGGLES:     parser.add_argument(f"--{e}", action="store_true")
    args = parser.parse_args()

    if base.local_rank == 0:
        base.log("Starting Flask host on rank 0", rank_0_only=True)
        app.run(host="localhost", port=args.port)
    else:
        while True:
            base.log(f"waiting for tasks")
            params = [{"stop": True}]
            dist.broadcast_object_list(params, src=0)
            if params[0].get("stop") is not None:
                base.log("Received exit signal, shutting down")
                return
            base.log(f"Received task")
            __handle_request_parallel(*params)
    return


@app.route("/<path>", methods=["GET", "POST"])
def handle_path(path):
    match path:
        # status
        case "initialize":
            return base.get_initialized_flask()
        case "applied":
            return __get_applied()
        case "progress":
            return base.get_progress_flask()

        # generation
        case "apply":
            return __apply_pipeline(request.json)
        case "generate":
            return __generate_image(request.json)
        case "offload":
            return __offload_modules()
        case "close":
            base.log("Received exit signal, shutting down")
            __close_pipeline()
            raise HostShutdown

        case _:
            return "", 404


asyncdiff_config = None
def __reset_asyncdiff(steps):
    global asyncdiff_config
    if asyncdiff_config.get("synced_percent") is not None and asyncdiff_config.get("synced_percent") > 0:
        __reset_asyncdiff_sync_state((steps * asyncdiff_config.get("synced_percent")) // 100)
    else:
        __reset_asyncdiff_sync_state(asyncdiff_config.get("synced_steps"))
    return


def __reset_asyncdiff_sync_state(synced_steps):
    global async_diff
    async_diff.reset_state(warm_up=synced_steps)
    return


def __handle_request_parallel(data):
    if data.get("pipeline_type") is not None:
        __apply_pipeline_parallel(data)
    elif data.get("seed") is not None:
        __generate_image_parallel(data)
    elif data.get("offload") is not None:
        __offload_modules_parallel()
    else:
        assert False, "Unknown data type"


def __apply_pipeline(data):
    dist.broadcast_object_list([data], src=0)
    response = __apply_pipeline_parallel(data)
    return response


def __generate_image(data):
    dist.broadcast_object_list([data], src=0)
    response = __generate_image_parallel(data)
    return jsonify(response)


def __close_pipeline():
    dist.broadcast_object_list([{"stop": True}], src=0)
    return


def __offload_modules():
    dist.broadcast_object_list([{"offload": True}], src=0)
    return __offload_modules_parallel()


def __offload_modules_parallel():
    return __move_pipe("cpu")


def __move_pipe_module(module, device):
    global base
    try:
        if vars(base.pipe)[module].device != torch.device(device=device):
            vars(base.pipe)[module] = vars(base.pipe)[module].to(device=device)
            if "cpu" in device:
                vars(base.pipe)[module] = vars(base.pipe)[module].cpu()
            return 0
        else:
            return -1
    except:
        return 1


def __move_pipe(device):
    global base

    # flag_sliced = False
    if base.applied.get("enable_vae_slicing") is not None and base.applied.get("enable_vae_slicing") == True:
        # flag_sliced = True
        # base.pipe.disable_vae_tiling()
        return '"enable_vae_slicing" is enabled - not offloading', 500

    flag_tiled = False
    if base.applied.get("enable_vae_tiling") is not None and base.applied.get("enable_vae_tiling") == True:
        # flag_tiled = True
        # base.pipe.disable_vae_tiling()
        return '"enable_vae_tiling" is enabled - not offloading', 500

    # TODO: make exception for moving module to cpu
    if base.applied.get("group_offload_config") is not None:
        return 'Group offloading active - not offloading', 500

    moved = []
    not_moved = []
    alr_moved = []
    base.pipe = base.pipe.to(device=device)
    if "cuda" in device: torch.cuda.set_device(device)
    for k, v in base.pipe.components.items():
        if v is not None:
            has_moved = __move_pipe_module(k, device)
            match has_moved:
                case 0: moved.append(k)
                case 1: not_moved.append(k)
                case -1: alr_moved.append(k)

    # if flag_sliced: base.pipe.enable_vae_slicing()
    # if flag_tiled: base.pipe.enable_vae_tiling()

    clean()
    dist.barrier()
    msg = f"Moved to {device}: {str(moved)}, Not moved to {device}: {str(not_moved)}, Already on {device}: {str(alr_moved)}"
    base.log(msg)
    return msg, 200


def __get_applied():
    global base
    return str(base.applied), 200

def __apply_pipeline_parallel(data):
    global async_diff, asyncdiff_config, base
    with torch.no_grad():
        result = base.setup_pipeline(data, backend_name="asyncdiff")
        if result[1] == 200:
            # set asyncdiff
            # asyncdiff
            ad_config = data.get("backend_config")
            assert ad_config is not None, "AsyncDiff configuration must be provided"
            asyncdiff_config = ad_config
            if base.pipeline_type in ["ad"]:
                ad_class = AsyncDiffAnimateDiff
            elif base.pipeline_type in ["flux"]:
                ad_class = AsyncDiffFlux
            elif base.pipeline_type in ["sd3"]:
                ad_class = AsyncDiffStableDiffusion3
            elif base.pipeline_type in ["wani2v", "want2v"]:
                ad_class = AsyncDiffWan
            elif base.pipeline_type in ["zimage"]:
                ad_class = AsyncDiffZImage
            else:
                ad_class = AsyncDiffStableDiffusion
            async_diff = ad_class(
                base.pipe,
                base.pipeline_type,
                model_n=asyncdiff_config.get("model_n"),
                stride=asyncdiff_config.get("stride"),
                time_shift=asyncdiff_config.get("time_shift")
            )
        return result


def __generate_image_parallel(data):
    global base

    if base.applied is None: __close_pipeline()
    data = base.prepare_inputs(data)

    with torch.no_grad():
        __move_pipe(f"cuda:{dist.get_rank()}")
        torch.cuda.reset_peak_memory_stats()
        __reset_asyncdiff(data["steps"])
        base.progress = 0

        # set scheduler
        if data["scheduler"] is not None:           base.set_scheduler(data["scheduler"])
        elif base.default_scheduler is not None:    base.pipe.scheduler = base.default_scheduler
        if data["latent"] is not None:              data["latent"] = base.process_input_latent(data["latent"])
        base.set_scheduler_timesteps(data["denoising_start"])

        # inference kwargs
        kwargs = base.get_inference_kwargs(data, can_use_compel=True)

        # inference
        dist.barrier()
        output = base.pipe(**kwargs)

        # clean up
        clean()

        # output
        if base.local_rank == 0:
            base.progress = 100
            if output is not None:
                if get_is_image_model(base.pipeline_type):
                    if base.pipeline_type in ["sdup"]:
                        output = output.images[0]
                    else:
                        output_images = output.images
                        if base.pipeline_type in ["flux"]: output_images = base.pipe._unpack_latents(output_images, data["height"], data["width"], base.pipe.vae_scale_factor)
                        images = base.convert_latent_to_image(output_images)
                        latents = base.convert_latent_to_output_latent(output_images)
                        return { "message": "OK", "output": pickle_and_encode_b64(images[0]), "latent": pickle_and_encode_b64(latents), "is_image": True }
                else:
                    output = output.frames[0]
                return { "message": "OK", "output": pickle_and_encode_b64(output), "is_image": False }
            else:
                return { "message": "No image from pipeline", "output": None, "is_image": False }


if __name__ == "__main__":
    __initialize_distributed_environment()
    __run_host()
