import base64
import pickle
import torch


from torchvision.transforms import ToPILImage, ToTensor


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
