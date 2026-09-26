# SPDX-License-Identifier: Apache-2.0
# Based on vLLM v0.29.0 Qwen GDN layer (copyright vLLM contributors).
"""Replace the post-convolution GDN boundary; retain model weights and scheduling."""

import atexit
import json
import os
from importlib.metadata import version
from pathlib import Path

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    fused_post_conv_prep,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)

from lightdelta.backend import GDNBackend
from lightdelta.config import BackendConfig

if version("vllm").split("+")[0] != "0.29.0":
    raise RuntimeError("LightDelta requires vLLM 0.29.0")

CONFIG = BackendConfig.from_file(os.environ["LIGHTDELTA_CONFIG"])
BACKEND = GDNBackend(CONFIG)
logger = init_logger("vllm.lightdelta")


@atexit.register
def _save_probe():
    if "LIGHTDELTA_PROBE_DIR" in os.environ and BACKEND.calls:
        path = Path(os.environ["LIGHTDELTA_PROBE_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        (path / f"paths-{os.getpid()}.json").write_text(json.dumps(dict(BACKEND.calls), indent=2))


@QwenGatedDeltaNetAttention.register_oot
class LightDeltaAttention(QwenGatedDeltaNetAttention):
    def _fused_gdn_decode_unsupported_reason(self, vllm_config):
        # The wheel need not contain vLLM's own fused operator: LightDelta ships its own.
        actual = (
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.norm.activation,
            vllm_config.model_config.dtype,
            self.get_state_dtype()[1],
        )
        if actual != (1, 16, 48, 128, 128, "silu", torch.bfloat16, torch.float32):
            return "LightDelta requires single-GPU Qwen3.8-27B BF16 and FP32 state"
        return None

    def __init__(self, config, vllm_config, prefix="", **kwargs):
        super().__init__(config, vllm_config, prefix, **kwargs)
        self.enable_fused_gdn_decode = True
        size = vllm_config.scheduler_config.max_num_seqs * (CONFIG.num_speculative_tokens + 1)
        self._accepted_ones = torch.ones(size, dtype=torch.int32, device=self.A_log.device)
        BACKEND.calls["registered_layers"] += 1
        logger.info_once(
            "LightDelta GDN: compute=%s, state_io=%s, prefill_io=%s",
            CONFIG.compute,
            CONFIG.state_io,
            CONFIG.prefill_io,
        )

    def _run_decode(self, qkv, a, b, gate, out, slots, cu, accepted):
        BACKEND(
            qkv,
            a,
            b,
            self.A_log,
            self.dt_bias,
            slots.contiguous(),
            cu,
            accepted,
            self.kv_cache[1],
            gate,
            self.norm.weight,
            out,
            self.layer_norm_epsilon,
        )
        if BACKEND.calls["decode" if slots.shape[1] == 1 else "verify"] == 1:
            _save_probe()

    def _prefill(self, qkv, a, b, gate, out, metadata):
        BACKEND.calls["prefill"] += 1
        if BACKEND.calls["prefill"] == 1:
            _save_probe()
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=qkv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=16,
            head_k_dim=128,
            head_v_dim=128,
            apply_l2norm=True,
            output_g_exp=False,
        )
        slots = metadata.prefill_state_indices
        valid = metadata.prefill_has_initial_state
        pool = self.kv_cache[1]
        if CONFIG.prefill_io == "fused_copy":
            initial = pool.new_empty((slots.numel(), *pool.shape[1:]))
            torch.ops.lightdelta.gather_state(pool, slots, valid, initial)
        else:
            initial = pool[slots]
            initial[~valid] = 0
        value, final = self.chunk_gated_delta_rule(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=metadata.prefill_query_start_loc,
            chunk_indices=metadata.chunk_indices,
            chunk_offsets=metadata.chunk_offsets,
            use_qk_l2norm_in_kernel=False,
        )
        if CONFIG.prefill_io == "fused_copy":
            torch.ops.lightdelta.scatter_state(final.contiguous(), slots, pool)
        else:
            pool[slots] = final
        self._rms_norm_gated_cuda(value.squeeze(0), gate, out)

    def _forward_core_fused_norm(self, mixed_qkv, b, a, output_gate, core_attn_out):
        context = get_forward_context()
        if context.attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return
        m = context.attn_metadata[self.prefix]
        n = m.num_actual_tokens
        qkv, a, b, gate = (x[:n] for x in (mixed_qkv, a, b, output_gate))
        conv = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2)
        weight = self.conv1d.weight.view(self.conv1d.weight.size(0), -1)

        if m.spec_sequence_masks is not None:
            pure_spec = m.num_prefills == 0 and m.num_decodes == 0
            sq, sa, sb, sg = (
                (qkv, a, b, gate)
                if pure_spec
                else tuple(x.index_select(0, m.spec_token_indx) for x in (qkv, a, b, gate))
            )
            count = m.num_spec_decodes
            slots = m.spec_state_indices_tensor[:count]
            cu = m.spec_query_start_loc[: count + 1]
            accepted = m.num_accepted_tokens[:count]
            sq = causal_conv1d_update(
                sq,
                conv,
                weight,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=slots[:, 0],
                num_accepted_tokens=accepted,
                query_start_loc=cu,
                max_query_len=slots.shape[1],
                validate_data=False,
            )
            sout = core_attn_out[:n] if pure_spec else torch.empty_like(sg)
            self._run_decode(sq, sa, sb, sg, sout, slots, cu, accepted)
            if pure_spec:
                return
            core_attn_out.index_copy_(0, m.spec_token_indx.long(), sout)
            qkv, a, b, gate = (x.index_select(0, m.non_spec_token_indx) for x in (qkv, a, b, gate))
            non_out = torch.empty_like(gate)
            BACKEND.calls["mixed"] += 1
        else:
            non_out = core_attn_out[:n]
            if m.num_prefills and m.num_decodes:
                BACKEND.calls["mixed"] += 1

        if m.num_prefills:
            qkv = causal_conv1d_fn(
                qkv.transpose(0, 1),
                weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv,
                has_initial_state=m.has_initial_state,
                cache_indices=m.non_spec_state_indices_tensor,
                query_start_loc=m.non_spec_query_start_loc,
                metadata=m,
            ).transpose(0, 1)
        elif m.num_decodes:
            qkv = causal_conv1d_update(
                qkv,
                conv,
                weight,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=m.non_spec_state_indices_tensor[: m.num_decodes],
                validate_data=False,
            )
        d = m.num_decode_tokens
        if m.num_decodes:
            count = m.num_decodes
            self._run_decode(
                qkv[:d],
                a[:d],
                b[:d],
                gate[:d],
                non_out[:d],
                m.non_spec_state_indices_tensor[:count, None],
                m.non_spec_query_start_loc[: count + 1],
                self._accepted_ones[:count],
            )
        if m.num_prefills:
            self._prefill(qkv[d:], a[d:], b[d:], gate[d:], non_out[d:], m)
        if m.spec_sequence_masks is not None:
            core_attn_out.index_copy_(0, m.non_spec_token_indx.long(), non_out)
