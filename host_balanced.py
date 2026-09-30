import argparse
import cache_dit
import logging
import os
import signal
import time
import torch
from DeepCache import DeepCacheSDHelper
from flask import Flask, request, jsonify


from modules.host_common import *
from modules.scheduler_config import *
from modules.utils import *


logging.getLogger("werkzeug").setLevel(logging.CRITICAL)
logging.getLogger("flask").setLevel(logging.CRITICAL)
logging.getLogger("werkzeug").disabled = True
logging.getLogger("flask").disabled = True
app = Flask(__name__)
base = None
balanced_config = None


def __initialize_environment():
    global base
    base = CommonHost()
    base.local_rank = 0
    base.set_logger()
    base.initialized = True
    return


def __run_host():
    global base
    parser = argparse.ArgumentParser()
    for k, v in GENERIC_HOST_ARGS.items():  parser.add_argument(f"--{k}", type=v, default=None)
    for e in GENERIC_HOST_ARGS_TOGGLES:     parser.add_argument(f"--{e}", action="store_true")
    args = parser.parse_args()

    torch._logging.set_logs(all=logging.CRITICAL)
    base.log(1, "ℹ️ Starting Flask host", rank_0_only=False)
    app.run(host="localhost", port=args.port, debug=False, use_reloader=False)
    return


@app.route("/<path>", methods=["GET", "POST"])
def handle_path(path):
    global base
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
            return __apply_pipeline_parallel(request.json)
        case "generate":
            return __generate_image_parallel(request.json)
        case "offload":
            return "Operation not supported by this host", 500
        case "sleep":
            return "Operation not supported by this host", 500
        case "close":
            base.log(0, "🛑 Received exit signal - shutting down", rank_0_only=False)
            base.close_pipeline()
            base = None
            os.kill(os.getpid(), signal.SIGTERM)
            #raise HostShutdown
            return "", 200
        case _:
            return "", 404


def __apply_pipeline_parallel(data):
    global base, balanced_config
    # Params:
    # deep_cache
    # deep_cache_interval
    # deep_cache_id
    # cache_dit
    config = data.get("backend_config")
    assert config is not None, "Configuration must be provided"
    with torch.no_grad():
        result = base.setup_pipeline(data, backend_name="balanced")
        if result[1] == 200:
            balanced_config = config
        return result


def __generate_image_parallel(data):
    global base, balanced_config
    data = base.prepare_inputs(data)

    with torch.inference_mode():
        torch.cuda.reset_peak_memory_stats()

        # inference kwargs
        kwargs = base.setup_inference(data, can_use_compel=False)

        # inference
        enable_deepcache = base.can_use_deepcache == True and balanced_config.get("deep_cache") == True
        enable_cache_dit = base.can_use_cachedit == True and balanced_config.get("cache_dit") == True
        with torch.autocast(device_type="cuda", dtype=base.infer_dtype):
            if enable_deepcache:
                helper = DeepCacheSDHelper(pipe=base.pipe)
                helper.set_params(cache_interval=balanced_config.get("deep_cache_interval"), cache_branch_id=balanced_config.get("deep_cache_id"))
                helper.enable()
                base.log(0, "ℹ️ DeepCache enabled", rank_0_only=False)
            start_time = time.perf_counter()
            output = base.pipe(**kwargs)
            end_time = time.perf_counter()
            base.log(0, f"⏱️ Processing time: {end_time - start_time:0.3f}")
            if enable_deepcache:
                helper.disable()
                base.log(0, "ℹ️ DeepCache disabled", rank_0_only=False)

        # clean up
        clean()

        # output
        if output is not None:
            if base.is_image_model:
                if base.pipeline_type in ["sdup"]:
                    output = output.images[0]
                else:
                    output_images = output.images
                    output_images = base.get_output_images(output_images)
                    flag = base.pipe.vae.device == torch.device("cpu")
                    if flag: base.pipe.vae = base.pipe.vae.to(device=output_images.device)
                    images = base.convert_latent_to_image(output_images)
                    latents = base.convert_latent_to_output_latent(output_images)
                    if flag: base.pipe.vae = base.pipe.vae.to(device="cpu")
                    return { "message": "OK", "output": pickle_and_encode_b64(images[0]), "latent": pickle_and_encode_b64(latents), "is_image": True }
            else:
                output = output.frames[0]
            return { "message": "OK", "output": pickle_and_encode_b64(output), "is_image": False }
        else:
            return { "message": "No image from pipeline", "output": None, "is_image": False }


if __name__ == "__main__":
    __initialize_environment()
    __run_host()
