# multigpu_diffusion

Python Flask hosts for multi-GPU Diffusion inferencing solutions.
(Uses HuggingFace Diffusers library.)


## Notes:
- Windows and macOS are (probably) not supported.
- This repo mainly exists for [multigpu_diffusion_comfyui](https://github.com/SlackinJack/multigpu_diffusion_comfyui), which provides the nodes to run everything here.


## Manual Usage:
- (Review and) run `setup.sh`. If you want to use a venv, ensure that it is active before running.
- For AsyncDiff host, run `torchrun --master_port={master_port} --nproc_per_node={n_gpus} host_asyncdiff.py --port={port}`
- For other hosts, run `python3 --host_{name}.py --port={port}`
- To interact with the hosts, GET/POST to localhost:{port}/{endpoint}. You can find the endpoints at each host's handle_path().


## Additional Resources:
- [AsyncDiff](https://github.com/czg1225/AsyncDiff)


## Test Environment:
- 4x Nvidia Tesla T4
- Ubuntu Server 26.04
- Python 3.14.4
