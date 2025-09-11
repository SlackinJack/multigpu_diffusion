import FilePatcher

root = "AsyncDiff/asyncdiff"

patches = [
    {
        "file_name": f"{root}/async_animate.py",
        "replace": [
            # Modify constructor to take pipeline_type as an argument
            # so that we can define it, instead of using the model name
            {
                "from": "def __init__(self, pipeline, model_n=2, stride=1, warm_up=1, time_shift=False):",
                "to":   "def __init__(self, pipeline, pipeline_type, model_n=2, stride=1, warm_up=1, time_shift=False):",
            },
            {
                "from": "self.pipe_id = pipeline.config._name_or_path",
                "to":   "self.pipe_id = pipeline_type",
            },
            # Increase torch timeout to 1 day
            {
                "from": 'dist.init_process_group("nccl")',
                "to":   """from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(days=1))""",
            },
        ],
    },
    {
        "file_name": f"{root}/async_sd.py",
        "replace": [
            # Modify constructor to take pipeline_type as an argument
            # so that we can define it, instead of using the model name
            {
                "from": "def __init__(self, pipeline, model_n=2, stride=1, warm_up=1, time_shift=False):",
                "to":   "def __init__(self, pipeline, pipeline_type, model_n=2, stride=1, warm_up=1, time_shift=False):",
            },
            {
                "from": "self.pipe_id = pipeline.config._name_or_path",
                "to":   "self.pipe_id = pipeline_type",
            },
            # Increase torch timeout to 1 day
            {
                "from": 'dist.init_process_group("nccl")',
                "to":   """from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(days=1))""",
            },
        ],
    },
    {
        "file_name": f"{root}/pipe_config.py",
        "replace": [
            # Change all model names to their corresponding pipeline_type
            {
                "from": "stabilityai/stable-diffusion-3-medium-diffusers",
                "to":   "sd3",
            },
            {
                "from": "stabilityai/stable-video-diffusion-img2vid-xt",
                "to":   "svd",
            },
            {
                "from": "stabilityai/stable-diffusion-2-1",
                "to":   "sd2",
            },
            {
                "from": "runwayml/stable-diffusion-v1-5",
                "to":   "sd1",
            },
            {
                "from": "pipe_id == \"stabilityai/stable-diffusion-xl-base-1.0\" or pipe_id == \"RunDiffusion/Juggernaut-X-v10\" or pipe_id == \"diffusers/stable-diffusion-xl-1.0-inpainting-0.1\"",
                "to":   "pipe_id == \"sdxl\"",
            },
            {
                "from": "emilianJR/epiCRealism",
                "to":   "ad",
            },
            {
                "from": "stabilityai/stable-diffusion-x4-upscaler",
                "to":   "sdup",
            },
            # Fix tuple @ line 355
            {
                "from": "unet.up_blocks[2]   ",
                "to": "unet.up_blocks[2],",
            },
        ],
    },
]

FilePatcher.patch(patches)
