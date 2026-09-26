"""The common post-convolution boundary used by tests, benchmarks and vLLM."""

from collections import Counter

import torch

from lightdelta import ops  # noqa: F401 -- dispatcher registration
from lightdelta.config import BackendConfig
from lightdelta.reference import GDNInputs, post_conv_gdn


class GDNBackend:
    def __init__(self, config: BackendConfig):
        config.validate()
        self.config = config
        self.calls = Counter()
        self._packed = {}

    def __call__(
        self, qkv, a, b, a_log, dt_bias, slots, cu, accepted, pool, gate, weight, out, eps=1e-6
    ):
        batch, width = slots.shape
        self.calls["decode" if width == 1 else "verify"] += 1
        if self.config.state_io == "indexed":
            torch.ops.lightdelta.decode(
                qkv, a, b, a_log, dt_bias, slots, cu, accepted, pool, gate, weight, out, eps
            )
            return

        # Both reference and fused-packed use these exact gather/scatter operations.
        key = (pool.device, batch, width)
        if key not in self._packed:
            scratch = pool.new_empty((1 + batch * width, *pool.shape[1:]))
            indices = torch.arange(
                1, 1 + batch * width, device=pool.device, dtype=torch.int32
            ).view(batch, width)
            ones = torch.ones(batch, device=pool.device, dtype=torch.int32)
            self._packed[key] = scratch, indices, ones
        scratch, compact, ones = self._packed[key]
        source = slots.gather(1, (accepted.clamp(1, width) - 1)[:, None].long()).flatten()
        valid = (accepted > 0) & (accepted <= width) & (source > 0)
        initial = scratch[1:].view(batch, width, *pool.shape[1:])[:, 0]
        torch.index_select(pool, 0, source.clamp_min(0).long(), out=initial)
        initial.masked_fill_(~valid[:, None, None, None], 0)
        lengths = cu[1:] - cu[:-1]
        offsets = torch.arange(width, device=pool.device)[None, :]
        live = offsets < lengths[:, None]

        if self.config.compute == "torch_reference":
            token_ids = cu[:-1, None] + offsets
            if width == 1:
                dense, ad, bd, gd = (x.unsqueeze(1) for x in (qkv, a, b, gate))
            else:
                ids = token_ids.clamp(0, qkv.shape[0] - 1).long()
                dense, ad, bd, gd = (x[ids] for x in (qkv, a, b, gate))
            q, k, v = dense.split((2048, 2048, 6144), dim=-1)
            result = post_conv_gdn(
                GDNInputs(
                    q.reshape(batch, width, 16, 128),
                    k.reshape(batch, width, 16, 128),
                    v.reshape(batch, width, 48, 128),
                    ad,
                    bd,
                    a_log,
                    dt_bias,
                    gd,
                    weight,
                ),
                initial,
                torch.where(valid, lengths, 0),
                save_checkpoints=width > 1,
                norm_eps=eps,
            )
            if width == 1:
                out.copy_(result.output[:, 0])
                states = result.final_state
            else:
                out.index_copy_(0, token_ids[live].long(), result.output[live])
                states = result.checkpoints[:, 1:].reshape(batch * width, *pool.shape[1:])
        else:
            indices = torch.where(valid[:, None], compact, 0)
            torch.ops.lightdelta.decode(
                qkv, a, b, a_log, dt_bias, indices, cu, ones, scratch, gate, weight, out, eps
            )
            states = scratch[1:]

        destinations = torch.where(live & valid[:, None] & (slots > 0), slots, 0).flatten()
        # Padding destinations preserve the reserved slot bit-for-bit. This avoids
        # dynamic boolean indexing and is also safe to capture in a CUDA graph.
        writeback = torch.where(destinations[:, None, None, None] > 0, states, pool[0])
        pool.index_copy_(0, destinations.long(), writeback)
