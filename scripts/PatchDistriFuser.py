import FilePatcher

root = "DistriFuser/distrifuser"
models = f"{root}/models"
modules = f"{root}/modules"

patches = [
    # models
    {
        "file_name": f"{models}/base_model.py",
        "replace": [
            {
                "from": "from distrifuser.modules",
                "to":   "from ..modules",
            },
        ],
    },
    {
        "file_name": f"{models}/distri_sdxl_unet_pp.py",
        "replace": [
            {
                "from": "from distrifuser.modules",
                "to":   "from ..modules",
            },
            {
                "from": "super(DistriUNetPP, self).__init__(model, distri_config)",
                "to": """super(DistriUNetPP, self).__init__(model, distri_config)


    def load_lora_adapter(self, pretrained_model_name_or_path_or_dict, **kwargs):
        return self.model.load_lora_adapter(pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict, **kwargs)


    def set_adapters(self, adapter_names, weights):
        return self.model.set_adapters(adapter_names=adapter_names, weights=weights)


    def unload_lora(self):
        return self.model.unload_lora()
""",
            },
        ],
    },
    {
        "file_name": f"{models}/distri_sdxl_unet_tp.py",
        "replace": [
            {
                "from": "from distrifuser.modules",
                "to":   "from ..modules",
            },
            {
                "from": "super(DistriUNetTP, self).__init__(model, distri_config)",
                "to": """super(DistriUNetTP, self).__init__(model, distri_config)


    def load_lora_adapter(self, pretrained_model_name_or_path_or_dict, **kwargs):
        return self.model.load_lora_adapter(pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict, **kwargs)


    def set_adapters(self, adapter_names, weights):
        return self.model.set_adapters(adapter_names=adapter_names, weights=weights)


    def unload_lora(self):
        return self.model.unload_lora()
""",
            },
        ],
    },
    # modules/pp
    {
        "file_name": f"{modules}/pp/attn.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    {
        "file_name": f"{modules}/pp/conv2d.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    {
        "file_name": f"{modules}/pp/groupnorm.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    # modules/tp
    {
        "file_name": f"{modules}/tp/attention.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    {
        "file_name": f"{modules}/tp/conv2d.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    # modules
    {
        "file_name": f"{modules}/base_module.py",
        "replace": [
            {
                "from": "from distrifuser.utils",
                "to":   "from ..utils",
            },
        ],
    },
    {
        "file_name": f"{root}/utils.py",
        "replace": [
            # Increase torch timeout to 1 day
            {
                "from": 'dist.init_process_group("nccl")',
                "to":   """from datetime import timedelta
            dist.init_process_group("nccl", timeout=timedelta(days=1))""",
            },
        ],
    },
]

FilePatcher.patch(patches)

