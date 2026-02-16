import argparse
import logging
import os
import signal
import torch
from flask import Flask, request, jsonify


from modules.host_common import *
from modules.scheduler_config import *
from modules.utils import *


app = Flask(__name__)
base = None


def __initialize_environment():
    global base
    base = CommonHost()
    base.local_rank = 0
    base.set_logger()
    base.initialized = True
    return


args = None
def __run_host():
    global args, base
    parser = argparse.ArgumentParser()
    # generic
    for k, v in GENERIC_HOST_ARGS.items():  parser.add_argument(f"--{k}", type=v, default=None)
    for e in GENERIC_HOST_ARGS_TOGGLES:     parser.add_argument(f"--{e}", action="store_true")
    args = parser.parse_args()

    torch._logging.set_logs(all=logging.CRITICAL)
    base.log("ℹ️ Starting Flask host")
    logging.getLogger('werkzeug').disabled = True
    app.run(host="localhost", port=args.port)
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
            return __offload_modules_parallel()
        case "close":
            base.log("🛑 Received exit signal - shutting down")
            base.close_pipeline()
            os.kill(os.getpid(), signal.SIGTERM)
            raise HostShutdown

        case _:
            return "", 404


def __offload_modules_parallel():
    return "Operation not supported by this host", 500


def __apply_pipeline_parallel(data):
    global base
    with torch.no_grad():
        return base.setup_pipeline(data, backend_name="balanced")


def __generate_image_parallel(data):
    global base

    data = base.prepare_inputs(data)

    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        base.progress = 0

        # inference kwargs
        kwargs = base.setup_inference(data, can_use_compel=False)

        # inference
        output = base.pipe(**kwargs)

        # clean up
        clean()

        # output
        base.progress = 100
        if output is not None:
            if get_is_image_model(base.pipeline_type):
                if base.pipeline_type in ["sdup"]:
                    output = output.images[0]
                else:
                    output_images = output.images
                    if base.pipeline_type in ["flux"]: output_images = base.pipe._unpack_latents(output_images, data["height"], data["width"], base.pipe.vae_scale_factor)
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
