import argparse
import logging
import os
import signal
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from flask import Flask, request, jsonify


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
asyncdiff_config = None
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
    parser.add_argument("--time_shift",     type=int,   default=0)
    parser.add_argument("--cache_step",     type=int,   default=1)
    # generic
    for k, v in GENERIC_HOST_ARGS.items():  parser.add_argument(f"--{k}", type=v, default=None)
    for e in GENERIC_HOST_ARGS_TOGGLES:     parser.add_argument(f"--{e}", action="store_true")
    args = parser.parse_args()

    torch._logging.set_logs(all=logging.CRITICAL)
    if base.local_rank == 0:
        base.log("ℹ️ Starting Flask host on rank 0", rank_0_only=True)
        logging.getLogger('werkzeug').disabled = True
        app.run(host="localhost", port=args.port)
    else:
        while True:
            base.log(f"⏳ Waiting for tasks")
            params = [{"stop": True}]
            # TODO: would be nice to make this non-blocking
            dist.broadcast_object_list(params, src=0)
            if params[0].get("stop") is not None:
                base.log("🛑 Received exit signal - shutting down")
                return
            elif params[0].get("sleep") is not None and params[0].get("time") is not None:
                t = params[0].get("time")
                base.log(f"💤 Received sleep signal - pausing for {t} seconds")
                time.sleep(int(t))
                base.log("⏰ Sleep finished")
            else:
                base.log(f"📋 Received task")
                __handle_request_parallel(*params)
    return


@app.route("/<path>", methods=["GET", "POST"])
def handle_path(path):
    match path:
        # status
        case "initialize":
            return base.get_initialized_flask()
        case "applied":
            return base.get_applied()
        case "progress":
            return base.get_progress_flask()

        # generation
        case "apply":
            return __apply_pipeline(request.json)
        case "generate":
            return __generate_image(request.json)
        case "offload":
            return __offload_modules()
        case "sleep":
            return __handle_sleep(request.json)
        case "close":
            base.log("🛑 Received exit signal - shutting down")
            dist.broadcast_object_list([{"stop": True}], src=0)
            base.close_pipeline()
            os.kill(os.getpid(), signal.SIGTERM)
            raise HostShutdown

        case _:
            return "", 404


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


def __handle_sleep(data):
    dist.broadcast_object_list([data], src=0)
    return "", 200


def __offload_modules():
    dist.broadcast_object_list([{"offload": True}], src=0)
    return __offload_modules_parallel()


def __offload_modules_parallel():
    return __move_pipe("cpu")


def __move_pipe_module(module, dev):
    global base
    try:
        mod = vars(base.pipe)[module]
        c_device = torch.device(mod.device)
        t_device = torch.device(dev)
        if c_device != t_device:
            vars(base.pipe)[module] = mod.to(device=t_device)
            if "cpu" in str(dev):
                vars(base.pipe)[module] = mod.cpu()
            elif "cuda" in str(dev):
                torch.cuda.set_device(tar_device)
                torch.cuda.synchronize()
            return 0
        else:
            return -1
    except:
        return 1


def __move_pipe(device):
    global base

    if base.applied.get("group_offload_config") is not None:
        return 'Group offloading active - not offloading', 500

    flag_vae_sliced = False
    if base.applied.get("enable_vae_slicing") is not None and base.applied.get("enable_vae_slicing") == True:
        flag_vae_sliced = True
        base.pipe.disable_vae_slicing()

    flag_vae_tiled = False
    if base.applied.get("enable_vae_tiling") is not None and base.applied.get("enable_vae_tiling") == True:
        flag_vae_tiled = True
        base.pipe.disable_vae_tiling()

    flag_attention_sliced = False
    if base.applied.get("enable_attention_slicing") is not None and base.applied.get("enable_attention_slicing") == True:
        flag_attention_sliced = True
        base.pipe.disable_attention_slicing()

    moved = []
    not_moved = []
    alr_moved = []
    for k, v in base.pipe.components.items():
        if v is not None:
            has_moved = __move_pipe_module(k, device)
            match has_moved:
                case 0: moved.append(k)
                case 1: not_moved.append(k)
                case -1: alr_moved.append(k)
    base.pipe = base.pipe.to(device=torch.device(device))

    if flag_vae_sliced: base.pipe.enable_vae_slicing()
    if flag_vae_tiled: base.pipe.enable_vae_tiling()
    if flag_attention_sliced: base.pipe.enable_attention_slicing()

    clean()
    dist.barrier()
    if ":" in device: device = device.split(":")[0]
    msg = f"""➡️ Moved pipeline components:
        Moved to {device}: {str(moved)}
        Not moved to {device}: {str(not_moved)}
        Already on {device}: {str(alr_moved)}"""
    base.log(msg, rank_0_only=True)
    return msg, 200


def __apply_pipeline_parallel(data):
    global async_diff, asyncdiff_config, base
    with torch.no_grad():
        result = base.setup_pipeline(data, backend_name="asyncdiff")
        if result[1] == 200:
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
                time_shift=asyncdiff_config.get("time_shift"),
                cache_step=asyncdiff_config.get("cache_step"),
            )
        return result


def __generate_image_parallel(data):
    global async_diff, asyncdiff_config, base

    data = base.prepare_inputs(data)

    with torch.no_grad():
        __move_pipe(f"cuda:{base.local_rank}")
        torch.cuda.reset_peak_memory_stats()
        warmup_steps = asyncdiff_config.get("synced_steps")
        # async_diff.reset_state(warm_up=warmup_steps)
        base.progress = 0

        def reset(c, d):
            nonlocal warmup_steps
            async_diff.reset_state(warm_up=warmup_steps)
            return

        def complete(c, d):
            base.log("🚀 AsyncDiff warmup completed", rank_0_only=True)
            return

        # inference kwargs
        callbacks = {}
        async_diff.reset_state(warm_up=warmup_steps)
        callbacks[warmup_steps] = complete
        kwargs = base.setup_inference(data, can_use_compel=True, callbacks=callbacks)

        # inference
        dist.barrier()
        with torch.inference_mode():
            output = base.pipe(**kwargs)
        dist.barrier()

        # clean up
        clean()

        base.progress = 100

        # output
        if base.local_rank == 0:
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
