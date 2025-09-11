import FilePatcher

root = "xDiT/xfuser"
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
        "file_name": f"{root}/model_executor/pipelines/pipeline_hunyuandit.py",
        "replace": [
            # Fix legacy output type error
            {
                "from": """        image_rotary_emb = get_2d_rotary_pos_embed(
            self.transformer.inner_dim // self.transformer.num_heads,
            grid_crops_coords,
            (grid_height, grid_width),
        )""",
                "to": """        image_rotary_emb = get_2d_rotary_pos_embed(
            self.transformer.inner_dim // self.transformer.num_heads,
            grid_crops_coords,
            (grid_height, grid_width),
            output_type="pt",
        )""",
            },
        ],
    },
    {
        "file_name": f"{root}/core/distributed/parallel_state.py",
        "replace": [
            # Increase torch timeout to 1 day
            {
                "from": """torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )""",
                "to": """from datetime import timedelta
        torch.distributed.init_process_group(
            timeout=timedelta(days=1),
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )""",
            },
        ],
    },
]

patches_flexible = [
    {
        "file_name": f"{root}/core/long_ctx_attention/ring/ring_flash_attn.py",
        "question": "Is your GPU based on Ampere or newer? (y/n)",
        "proceed": ["n", "no"],
        "replace": [
            # only use flash attention when its supported
            {
                "from": """block_out, block_lse = fn(
                q,
                key,
                value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )""",
                "to": """block_out, block_lse = fn(
                q,
                key,
                value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
                op_type="flash" if flash_attn is not None else "efficient",
            )""",
            }
        ],
    },
]

FilePatcher.patch_any(root, target_type, patches_any)
FilePatcher.patch(patches)
FilePatcher.patch_flexible(patches_flexible)
