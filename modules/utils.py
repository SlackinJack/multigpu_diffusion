import base64
import gc
import inspect
import json
import pickle
import torch


from torchvision.transforms import ToPILImage, ToTensor


def clean_override_function(klass, target_func_name):
    module = inspect.getmodule(klass)
    if module is not None:
        if hasattr(module, f'old_{target_func_name}'):
            setattr(module, target_func_name, getattr(module, f'old_{target_func_name}'))
            delattr(module, f'old_{target_func_name}')
        if hasattr(module, f'new_{target_func_name}'):
            delattr(module, f'new_{target_func_name}')
    return


def override_function(klass, target_func_name, func):
    module = inspect.getmodule(klass)
    clean_override_function(klass, target_func_name)
    old_func = getattr(module, target_func_name, None)
    setattr(module, f'old_{target_func_name}', old_func)
    setattr(module, target_func_name, func)
    return


def get_function_from_class(klass, target_func_name):
    module = inspect.getmodule(klass)
    return getattr(module, target_func_name, None)


def get_torch_type(t):
    match t:
        case "bf16":    return torch.bfloat16
        case "fp8":     return torch.float8_e4m3fn
        case "fp16":    return torch.float16
        case "fp32":    return torch.float32
        case "int1":    return torch.uint1
        case "int2":    return torch.uint2
        case "int3":    return torch.uint3
        case "int4":    return torch.uint4
        case "int5":    return torch.uint5
        case "int6":    return torch.uint6
        case "int7":    return torch.uint7
        case "int8":    return torch.uint8
        case "int16":   return torch.uint16
        case "int32":   return torch.uint32
        case "bool":    return torch.bool
        case _:         return None


def clean():
    torch.cuda.memory.empty_cache()
    gc.collect()
    return


def normalize_latent(x, max_val=3.0):
    max_val = torch.tensor(max_val)
    x = x.detach().clone()
    for i in range(x.shape[0]):
        x[[i], :] = torch.sqrt(max_val**2 - 1) * x[[i], :] / torch.sqrt(torch.add(x[[i], :]**2, max_val**2 - 1))
        for chl in range(4):
            if x[i, chl, :, :].std() > 1.0:
                x[i, chl, :, :] /= x[i, chl, :, :].std()
    return x


def add_alpha_to_latent(latents):
    if latents.shape[1] == 3:
        alpha = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], device=latents.device)
        latents = torch.cat((latents, alpha), dim=1)
    return latents


def remove_alpha_from_latent(latents):
    if latents.shape[1] == 4:
        latents = latents[:, :3, :, :]
    return latents


def decode_b64_and_unpickle(b64):
    if b64 is None: return None
    out = base64.b64decode(b64)
    out = pickle.loads(out)
    return out


def pickle_and_encode_b64(obj):
    if obj is None: return None
    out = pickle.dumps(obj)
    out = base64.b64encode(out).decode('utf-8')
    return out


def convert_b64_to_nhwc_tensor(b64):
    if b64 is None: return None
    im2 = decode_b64_and_unpickle(b64)
    tensor_image = ToTensor()(im2)                      # CHW
    tensor_image = tensor_image.unsqueeze(0)            # CHW -> NCHW
    tensor_image = tensor_image.permute(0, 2, 3, 1)     # NCHW -> NHWC
    return tensor_image


def convert_image_to_hwc_tensor(image):
    if image is None: return None
    tensor = ToTensor()(image)          # CHW
    tensor = tensor.permute(1, 2, 0)    # CHW -> HWC
    return tensor


def convert_tensor_to_b64(tensor):
    if tensor is None: return None
    im = tensor.permute(2, 0, 1)        # HWC -> CHW
    im = ToPILImage()(im)
    b64_image = pickle_and_encode_b64(im)
    return b64_image


def format_json(str_or_dict_in, indent=4, indent_all=False):
    if isinstance(str_or_dict_in, dict): str_or_dict_in = json.dumps(str_or_dict_in)
    d = json.loads(str_or_dict_in)
    d = json.dumps(d, indent=indent)
    if indent_all == True:
        d2 = d.split("\n")
        out = ""
        for d3 in d2:
            if len(out) != 0:
                out += " "*indent
            out += d3 + "\n"
        d = out[::-1].replace("\n","",1)[::-1]
    return d
