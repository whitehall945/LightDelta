"""Native PyTorch dispatcher registration."""

from importlib.util import find_spec

import torch

torch.ops.load_library(find_spec("lightdelta._C").origin)


@torch.library.register_fake("lightdelta::decode")
def _decode_fake(qkv, a, b, a_log, dt_bias, indices, cu, accepted, state, gate, weight, out, eps):
    return None


@torch.library.register_fake("lightdelta::gather_state")
def _gather_fake(pool, slots, valid, out):
    return None


@torch.library.register_fake("lightdelta::scatter_state")
def _scatter_fake(packed, slots, pool):
    return None
