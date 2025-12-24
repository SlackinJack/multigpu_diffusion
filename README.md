# multigpu_diffusion

Python Flask hosts for distributed parallel Diffusion inferencing solutions.

## Model Compatibility Table:
*Note: This table only indicates what is supported in **this** project - it is not indicative of what is supported in the software being implemented.*

### AsyncDiff:
| Model                     | Type Name | LoRA | ControlNet | IPAdapter | Quantization | Compile | CPU Offloading |
|---------------------------|-----------|------|------------|-----------|--------------|---------|----------------|
| AnimateDiff               | ad        | ❌    | ✅          | ✅         | ✅            | ✅❓      | ❌              |
| Stable Diffusion 1.5      | sd1       | ✅❓   | ✅❓         | ✅❓        | ✅❓           | ✅❓      | ❌              |
| Stable Diffusion 2        | sd2       | ✅❓   | ✅❓         | ✅❓        | ✅❓           | ✅❓      | ❌              |
| Stable Diffusion 3        | sd3       | ✅❓   | ✅❓         | ✅❓        | ✅❓           | ✅❓      | ❌              |
| Stable Diffusion Upscaler | sdup      | ❌    | ❌          | ❌         | ✅❓           | ✅❓      | ❌              |
| Stable Diffusion XL       | sdxl      | ✅    | ✅          | ✅         | ✅            | ✅❓      | ❌              |
| Stable Video Diffusion    | svd       | ❌    | ❌          | ❌         | ✅❓           | ✅❓      | ❌              |


### xDiT:
| Model              | Type Name | LoRA   | ControlNet | IPAdapter | Quantization | Compile | CPU Offloading |
|--------------------|-----------|--------|------------|-----------|--------------|---------|----------------|
| FLUX.1             | flux      | ✅❓     | ❌          | ✅         | ✅            | ✅       | ✅❓             |
| HunYuan            | hy        | ✅❓     | ❌          | ✅❓        | ✅❓           | ✅❓      | ✅❓             |
| PixArt Alpha       | pixa      | ✅❓     | ❌          | ✅❓        | ✅❓           | ✅❓      | ✅❓             |
| PixArt Sigma       | pixs      | ✅❓     | ❌          | ✅❓        | ✅❓           | ✅❓      | ✅❓             |
| Stable Diffusion 3 | sd3       | ✅❓     | ❌          | ✅❓        | ✅            | ✅       | ✅❓             |


*❓: Untested*


## Host Arguments and Switches:
| Argument (--argument="value") | Type       | Usage                                              |
|-------------------------------|------------|----------------------------------------------------|
| height                        | int        | Image height                                       |
| width                         | int        | Image width                                        |
| warm_up_steps                 | int        | Number of warm up steps                            |
| port                          | int        | Port to run Flask server                           |
| type                          | str        | Model type name                                    |
| variant                       | str        | Torch variant                                      |
| scheduler                     | str        | Scheduler                                          |
| quantize_to                   | str        | Quantize the model                                 |
| checkpoint                    | str        | Checkpoint path                                    |
| gguf_model                    | str        | GGUF model path (usually for the UNet/Transformer) |
| motion_module                 | str        | Motion module path                                 |
| motion_adapter                | str        | Motion adapter path                                |
| control_net                   | str        | ControlNet path                                    |
| vae                           | str        | VAE path                                           |
| lora                          | str (dict) | LoRAs, in the format of { "path": lora_scale }     |
| ip_adapter                    | str (dict) | IPAdapter, in the format of { "path": ip_scale }   |

| Switches (--switch)           | Usage                                |
|-------------------------------|--------------------------------------|
| compel                        | Enable Compel                        |
| enable_vae_tiling             | Enable VAE tiling                    |
| enable_vae_slicing            | Enable VAE slicing                   |
| xformers_efficient            | Enable xFormers efficient operations |
| enable_model_cpu_offload      | Enable CPU model offloading          |
| enable_sequential_cpu_offload | Enable CPU sequential offloading     |
| compile_unet                  | Compile the UNet                     |
| compile_vae                   | Compile the VAE                      |
| compile_text_encoder          | Compile the Text Encoder             |

## Usage:
Check out [multigpu_diffusion_comfyui](https://github.com/SlackinJack/multigpu_diffusion_comfyui) and [multigpu_diffusion_localai](https://github.com/SlackinJack/multigpu_diffusion_localai) for running implementations.


## Currently Implemented Software:
- [AsyncDiff](https://github.com/czg1225/AsyncDiff)
- [xDiT](https://github.com/xdit-project/xDiT)
