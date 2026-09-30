# multigpu_diffusion

Python Flask hosts for multi-GPU Diffusion inferencing solutions.
(Uses HuggingFace Diffusers library.)


## Notes:
- Windows and macOS are not (and probably never will be) supported.
- This repo mainly exists for [multigpu_diffusion_comfyui](https://github.com/SlackinJack/multigpu_diffusion_comfyui), which provides ComfyUI nodes to run everything here.


## Usage:
- (Review and) run `setup.sh`. If you want to use a venv, ensure that it is active before running.
- For AsyncDiff host, run `torchrun --master_port={master_port} --nproc_per_node={n_gpus} host_asyncdiff.py --port={port}`
- For other hosts, run `python3 --host_{name}.py --port={port}`
- To interact with the hosts, GET/POST to localhost:{port}/{endpoint}. You can find the endpoints at each host's handle_path().


## Hosts:
| Host Name | Description                                                                       |
|    ---    |     ---                                                                           |
| AsyncDiff | Accelerates inference by caching individual model components across GPUs.         |
| Balanced  | Splits pipeline components so that they fit into VRAM (device_map="balanced").    |
|  Single   | Inference on a single GPU. Set the device via cuda_visible_devices.               |


## Additional Resources:
- [AsyncDiff](https://github.com/czg1225/AsyncDiff)


## Test Environment:
- 4x Nvidia Tesla T4
- Ubuntu Server 26.04
- Python 3.14.4
