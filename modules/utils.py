import base64
import pickle
import torch


from torchvision.transforms import ToPILImage, ToTensor


def decode_b64_and_unpickle(b64):
    out = base64.b64decode(b64)
    out = pickle.loads(out)
    return out


def pickle_and_encode_b64(obj):
    out = pickle.dumps(obj)
    out = base64.b64encode(out).decode('utf-8')
    return out


def get_torch_dtype_from_type(t):
    match t:
        case "bf16":    torch_dtype = torch.bfloat16
        case "fp16":    torch_dtype = torch.float16
        case _:         torch_dtype = torch.float32
    return torch_dtype


def convert_b64_to_nhwc_tensor(b64):
    im2 = decode_b64_and_unpickle(b64)
    tensor_image = ToTensor()(im2)                      # CHW
    tensor_image = tensor_image.unsqueeze(0)            # CHW -> NCHW
    tensor_image = tensor_image.permute(0, 2, 3, 1)     # NCHW -> NHWC
    return tensor_image


def convert_image_to_hwc_tensor(image):
    tensor = ToTensor()(image)          # CHW
    tensor = tensor.permute(1, 2, 0)    # CHW -> HWC
    return tensor


def convert_tensor_to_b64(tensor):
    im = tensor.permute(2, 0, 1)        # HWC -> CHW
    im = ToPILImage()(im)
    b64_image = pickle_and_encode_b64(im)
    return b64_image
