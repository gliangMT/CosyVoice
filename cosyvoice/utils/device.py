from contextlib import nullcontext

import torch


def is_musa_available():
    return hasattr(torch, 'musa') and torch.musa.is_available()


def is_cuda_available():
    return torch.version.cuda is not None and torch.cuda.is_available()


def get_available_device():
    if is_musa_available():
        return torch.device('musa')
    if is_cuda_available():
        return torch.device('cuda')
    return torch.device('cpu')


def is_gpu_available():
    return get_available_device().type in ('cuda', 'musa')


def get_device_module(device):
    if device.type == 'musa':
        return torch.musa
    if device.type == 'cuda':
        return torch.cuda
    return None


def stream_context(device):
    device_module = get_device_module(device)
    if device_module is None:
        return nullcontext()
    return device_module.stream(device_module.Stream(device))


def empty_cache_and_sync(device):
    device_module = get_device_module(device)
    if device_module is None:
        return
    if hasattr(device_module, 'empty_cache'):
        device_module.empty_cache()
    device_module.current_stream().synchronize()
