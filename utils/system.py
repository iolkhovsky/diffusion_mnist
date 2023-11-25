import platform
import torch


def get_available_device(verbose=True):
    has_mps = torch.backends.mps.is_built()
    has_cuda = torch.backends.cuda.is_built()
    exec_device = torch.device('cpu')
    if has_mps:
        exec_device = torch.device('mps')
    if has_cuda:
        exec_device = torch.device('cuda')
    if verbose:
        print(f'Platform: {platform.system()}')
        print(f'Release: {platform.release()}')
        print(f'MPS available: {has_mps}')
        print(f'CUDA available: {has_cuda}')
        print(f'Selected device: {exec_device}')
    return exec_device
