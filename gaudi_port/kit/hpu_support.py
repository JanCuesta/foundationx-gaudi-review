"""Minimal FP32 helpers for the experimental Foundation X+ Gaudi port."""
from contextlib import nullcontext
import torch
try:
    import habana_frameworks.torch  # noqa: F401 - registers HPU backend
except ImportError:
    pass  # permits syntax / CPU-only import testing outside Gaudi


def autocast(*args, **kwargs):
    # FP32 first: avoiding CUDA AMP and its GradScaler dependency.
    return nullcontext()


def sync():
    hpu = getattr(torch, 'hpu', None)
    if hpu is not None and hpu.is_available():
        hpu.synchronize()


def empty_cache():
    hpu = getattr(torch, 'hpu', None)
    if hpu is not None:
        func = getattr(hpu, 'empty_cache', None)
        if callable(func):
            func()


def max_memory_allocated():
    hpu = getattr(torch, 'hpu', None)
    if hpu is not None:
        func = getattr(hpu, 'max_memory_allocated', None)
        if callable(func):
            return func()
    return 0


def set_device(idx):
    # These jobs use one HPU and no distributed process group.
    if idx not in (0, '0', None):
        raise ValueError('Only single-device HPU jobs are supported by this port')


class NativeScaler:
    """FP32 drop-in for timm NativeScaler, with no CUDA GradScaler."""
    def __init__(self):
        self._state = {}

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None,
                 create_graph=False, need_update=True, update_grad=True, **kwargs):
        loss.backward(create_graph=create_graph)
        if clip_grad is not None and parameters is not None:
            if clip_mode != 'norm':
                raise NotImplementedError('HPU FP32 scaler supports norm clipping only')
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        if need_update and update_grad:
            optimizer.step()

    def state_dict(self):
        return self._state

    def load_state_dict(self, state_dict):
        self._state = dict(state_dict or {})


class GradScaler:
    """FP32 replacement for a disabled CUDA GradScaler in the DINO engine."""
    def __init__(self, *args, **kwargs):
        pass

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        return optimizer.step()

    def update(self, *args, **kwargs):
        return None
