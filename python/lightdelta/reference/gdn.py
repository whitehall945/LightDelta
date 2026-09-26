"""Unfused post-convolution GDN, vectorized over requests and heads.

Mathematical reference: Transformers b22fe462, Qwen3.5 recurrent gated delta rule.
State orientation and post-recurrence rounding follow vLLM's fused GDN boundary.
Precision and state layout are specified in docs/gdn-contract.md.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class GDNInputs:
    q: Tensor  # [B, T, Hk, K], after causal convolution and activation
    k: Tensor
    v: Tensor  # [B, T, Hv, V]
    a: Tensor  # [B, T, Hv], raw decay projection
    b: Tensor  # [B, T, Hv], raw update projection
    a_log: Tensor  # [Hv]
    dt_bias: Tensor  # [Hv]
    gate: Tensor  # [B, T, Hv, V], raw output gate
    norm_weight: Tensor  # [V], shared across heads

    def validate(self) -> None:
        if self.q.ndim != 4 or self.v.ndim != 4:
            raise ValueError("q and v must be [batch, tokens, heads, dimension]")
        batch, tokens, key_heads, key_dim = self.q.shape
        value_heads, value_dim = self.v.shape[-2:]
        if min(batch, tokens, key_heads, key_dim, value_heads, value_dim) <= 0:
            raise ValueError("Input dimensions must be positive; padding uses lengths=0")
        if value_heads % key_heads:
            raise ValueError("Value heads must be divisible by key heads")
        expected = {
            "k": self.q.shape,
            "v": (batch, tokens, value_heads, value_dim),
            "a": (batch, tokens, value_heads),
            "b": (batch, tokens, value_heads),
            "a_log": (value_heads,),
            "dt_bias": (value_heads,),
            "gate": self.v.shape,
            "norm_weight": (value_dim,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != tuple(shape):
                raise ValueError(f"{name} must have shape {tuple(shape)}")
        for name in self.__dataclass_fields__:
            tensor = getattr(self, name)
            if tensor.device != self.q.device or tensor.dtype not in (
                torch.float32,
                torch.bfloat16,
            ):
                raise ValueError(f"{name} must be FP32/BF16 on the same device as q")
        for name in ("k", "v", "a", "b", "gate"):
            if getattr(self, name).dtype != self.q.dtype:
                raise ValueError(f"{name} must have the same dtype as q")
        if self.a_log.dtype != torch.float32:
            raise ValueError("a_log must be FP32")


@dataclass(frozen=True)
class GDNResult:
    output: Tensor  # [B, T, Hv, V], input dtype
    final_state: Tensor  # [B, Hv, V, K], FP32
    checkpoints: Tensor  # [B, T+1, Hv, V, K] or [B, 0, Hv, V, K]


def post_conv_gdn(
    inputs: GDNInputs,
    initial_state: Tensor,
    lengths: Tensor,
    *,
    save_checkpoints: bool = False,
    norm_eps: float = 1e-6,
) -> GDNResult:
    """Compute without modifying inputs; checkpoint 0 is the initial state.

    `lengths` values must be validated by the metadata builder before execution.
    No device-to-host reads or per-request loops occur in this function.
    """
    inputs.validate()
    if norm_eps <= 0:
        raise ValueError("norm_eps must be positive")
    batch, tokens, key_heads, key_dim = inputs.q.shape
    value_heads, value_dim = inputs.v.shape[-2:]
    if initial_state.shape != (batch, value_heads, value_dim, key_dim):
        raise ValueError("initial_state must be [B, Hv, V, K]")
    if initial_state.dtype != torch.float32 or initial_state.device != inputs.q.device:
        raise ValueError("initial_state must be FP32 on the input device")
    if lengths.shape != (batch,) or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must be an int32/int64 vector of size B")
    if lengths.device != inputs.q.device:
        raise ValueError("lengths must be on the input device")

    q, k = inputs.q.float(), inputs.k.float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    ratio = value_heads // key_heads
    q = q.repeat_interleave(ratio, dim=2) * key_dim**-0.5
    k = k.repeat_interleave(ratio, dim=2)
    decay = (-inputs.a_log.exp() * F.softplus(inputs.a.float() + inputs.dt_bias.float())).exp()
    beta = inputs.b.float().sigmoid()
    v = inputs.v.float()
    state = initial_state
    outputs = []
    history = [state] if save_checkpoints else []
    for t in range(tokens):
        live = lengths > t
        decayed = state * decay[:, t, :, None, None]
        prediction = (decayed * k[:, t, :, None, :]).sum(-1)
        delta = (v[:, t] - prediction) * beta[:, t, :, None]
        updated = decayed + delta[..., None] * k[:, t, :, None, :]
        state = torch.where(live[:, None, None, None], updated, state)
        raw = (state * q[:, t, :, None, :]).sum(-1).to(inputs.q.dtype).float()
        # Match vLLM: round recurrence output, then norm/weight/SiLU in FP32.
        normalized = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + norm_eps)
        output = normalized * inputs.norm_weight.float() * F.silu(inputs.gate[:, t].float())
        outputs.append(torch.where(live[:, None, None], output, 0).to(inputs.q.dtype))
        if save_checkpoints:
            history.append(state)
    checkpoints = (
        torch.stack(history, dim=1)
        if save_checkpoints
        else initial_state.new_empty((batch, 0, value_heads, value_dim, key_dim))
    )
    return GDNResult(torch.stack(outputs, dim=1), state, checkpoints)
