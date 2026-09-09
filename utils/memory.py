# Copied from https://github.com/lllyasviel/FramePack/tree/main/demo_utils
# Apache-2.0 License
# By lllyasviel

import gc
import torch
from utils.console import *
from utils.distributed import get_rank
from utils.distributed import is_node_main
from configs import debug_options

cpu = torch.device('cpu')
gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
gpu_complete_modules = []


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return


def get_cuda_free_memory_gb(device=None):
    if device is None:
        device = gpu

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats['active_bytes.all.current']
    bytes_reserved = memory_stats['reserved_bytes.all.current']
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024 ** 3)


def log_gpu_memory(stage: str, device=None, rank=0, rank0_only=True):
    if not debug_options.LOG_GPU_MEMORY:
        return
    if rank0_only and rank != 0:
        return
    """Log GPU memory usage at a given training stage."""
    free_gb = get_cuda_free_memory_gb(device)
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    used_gb = total_gb - free_gb
    log(f"[Rank {rank}] [GPU Memory][{stage}] Used: {used_gb:.2f} GB | Free: {free_gb:.2f} GB | Total: {total_gb:.2f} GB")


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    log(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) <= preserved_memory_gb:
            gc.collect(); torch.cuda.empty_cache()
            return

        if hasattr(m, 'weight'):
            m.to(device=target_device)

    model.to(device=target_device)
    gc.collect(); torch.cuda.empty_cache()
    return


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    log(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) >= preserved_memory_gb:
            gc.collect(); torch.cuda.empty_cache()
            return

        if hasattr(m, 'weight'):
            m.to(device=cpu)

    model.to(device=cpu)
    gc.collect(); torch.cuda.empty_cache()
    return


def unload_complete_models(*args):
    for m in gpu_complete_modules + list(args):
        m.to(device=cpu)
        log(f'Unloaded {m.__class__.__name__} as complete.')

    gpu_complete_modules.clear()
    gc.collect(); torch.cuda.empty_cache()
    return


def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    log(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)
    return


def host_mem_gb():
    """(VmRSS, VmHWM_peak, VmPin_pinned) in GiB from /proc/self/status; -1 where unavailable.
    VmPin tracks cudaHostRegister'd page-locked memory -> directly surfaces a KV-offload pinned-cache leak;
    VmRSS is the cgroup-OOM-relevant anon+file RSS; VmHWM is the per-process peak (catches the ckpt gather)."""
    out = {'VmRSS': -1.0, 'VmHWM': -1.0, 'VmPin': -1.0}
    try:
        with open('/proc/self/status') as f:
            for line in f:
                key = line.split(':', 1)[0]
                if key in out:
                    out[key] = int(line.split()[1]) / (1024 ** 2)  # kB -> GiB
    except Exception:
        pass
    return out['VmRSS'], out['VmHWM'], out['VmPin']


def empty_cache_with_log(step: int, reason: str = ''):
    """gc.collect() then torch.cuda.empty_cache(), logging before/after GPU AND host memory (rank 0 only)."""
    rank = get_rank()
    gc.collect()
    before_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    before_rsrv = torch.cuda.memory_reserved() / (1024 ** 3)
    torch.cuda.empty_cache()
    after_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    after_rsrv = torch.cuda.memory_reserved() / (1024 ** 3)
    rss, hwm, pin = host_mem_gb()
    if is_node_main():
        log(yellow(
            f'[step {step} rank {rank}] empty_cache ({reason}): '
            f'alloc {before_alloc:.2f}→{after_alloc:.2f} GiB, '
            f'reserved {before_rsrv:.2f}→{after_rsrv:.2f} GiB '
            f'(freed {before_rsrv - after_rsrv:.2f} GiB reserved) | '
            f'HOST rss {rss:.1f} GiB, peak {hwm:.1f} GiB, pinned {pin:.1f} GiB'
        ))
