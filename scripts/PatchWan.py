import FilePatcher

root = "Wan2_1/wan"
target_type = "*.py"

patches_any = [
    {
        "replace": [
            {
                "from": "from xfuser.",
                "to":   "from xDiT.xfuser.",
            },
            {
                "from": "from xfuser ",
                "to":   "from xDiT.xfuser ",
            },
            {
                "from": "import xfuser.",
                "to":   "import xDiT.xfuser.",
            },
            {
                "from": "import xfuser ",
                "to":   "import xDiT.xfuser ",
            },
        ],
    },
]

patches = [
    {
        "file_name": f"{root}/text2video.py",
        "replace": [
            # add gguf, dtype arg
            {
                "from": """def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):""",
                "to": """def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        torch_dtype=torch.bfloat16,
        gguf=None,
    ):""",
            },
            # add gguf, tweak model loader
            {
                "from": """logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)""",
                "to": """logging.info(f"Creating WanModel from {checkpoint_dir}")
        if gguf is not None:
            self.model = WanModel.from_pretrained(
                checkpoint_dir,
                gguf_file=gguf,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True)
        else:
            self.model = WanModel.from_pretrained(
                checkpoint_dir,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True)""",
            },
        ],
    },
]

patches_flexible = [
    {
        "file_name": f"{root}/distributed/xdit_context_parallel.py",
        "question": "Is your GPU based on Ampere or newer? (y/n)",
        "proceed": ["n", "no"],
        "replace": [
            # change this to call pytorch attention by default
            # so that we can use the op_type arg
            {
                "from": """x = xFuserLongContextAttention()(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        window_size=self.window_size)""",
                "to": """from yunchang.kernels import AttnType
    x = xFuserLongContextAttention(attn_type=AttnType.TORCH)(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        window_size=self.window_size)""",
            },
        ],
    },
]

FilePatcher.patch_any(root, target_type, patches_any)
FilePatcher.patch(patches)
FilePatcher.patch_flexible(patches_flexible)
