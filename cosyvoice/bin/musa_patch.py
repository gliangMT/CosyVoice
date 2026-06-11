import torch


def musa_patch_simple():
    torch.cuda.set_device = torch.musa.set_device
    torch.cuda.is_available = torch.musa.is_available
    torch.cuda.current_device = torch.musa.current_device
    torch.nn.Module.cuda = torch.nn.Module.musa
    torch.cuda.amp.autocast = torch.musa.amp.autocast
    torch.cuda.amp.GradScaler = torch.musa.amp.GradScaler
    torch.cuda.current_stream = torch.musa.current_stream
    torch.cuda.Stream = torch.musa.Stream
    torch.cuda.stream = torch.musa.stream
    torch.cuda.manual_seed_all = torch.musa.manual_seed_all

def _get_available_device_type():
    if torch.cuda.is_available():
        return "musa"
    if torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
        return "xpu"
    if hasattr(torch, "mtia") and torch.mtia.is_available():
        return "mtia"
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, custom_backend_name, None)
    if custom_device_mod and custom_device_mod.is_available():
        return custom_backend_name
    # add more available device types here
    return None

def patch_torch_device():
    if getattr(torch.device, "_musa_patched", False):
        return

    original_device = torch.device

    def hook_cuda_device(device):
        if isinstance(device, str) and device.startswith("cuda"):
            return device.replace("cuda", "musa")
        if isinstance(device, original_device) and device.type == "cuda":
            return original_device("musa", device.index)
        return device

    class _PatchedTorchDeviceMeta(type):
        def __call__(cls, *args, **kwargs):
            if args:
                args = (hook_cuda_device(args[0]),) + args[1:]
            return original_device(*args, **kwargs)

        def __instancecheck__(cls, instance):
            return isinstance(instance, original_device)

        def __subclasscheck__(cls, subclass):
            return subclass is original_device or issubclass(subclass, original_device)

    class _PatchedTorchDevice(metaclass=_PatchedTorchDeviceMeta):
        pass

    _PatchedTorchDevice._musa_patched = True
    _PatchedTorchDevice._musa_original_device = original_device
    torch.device = _PatchedTorchDevice

def apply_musa_patch_simple():
    musa_patch_simple()
    patch_torch_device()
    torch._utils._get_available_device_type = _get_available_device_type


apply_musa_patch_simple()
