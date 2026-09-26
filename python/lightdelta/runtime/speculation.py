"""LightDelta-owned fixed-depth greedy MTP runtime.

The scheduler, target model and cache allocator remain vLLM services. Proposal,
stateful draft-step orchestration and greedy verification are implemented here.
The implementation intentionally supports the measured Qwen3.8-27B path only.
"""

from typing import Any

import torch
from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
    init_attn_backend,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


class LightDeltaSpeculator:
    """A compact greedy MTP coordinator for Qwen3.8-27B."""

    supports_mm_inputs = False
    draft_logits = None
    draft_token_confidence_probs = None

    def __init__(self, vllm_config, device: torch.device):
        spec = vllm_config.speculative_config
        if spec is None or spec.method != "mtp":
            raise ValueError("LightDeltaSpeculator requires method=mtp")
        if spec.num_speculative_tokens not in (1, 2, 3):
            raise ValueError("LightDeltaSpeculator supports one through three draft tokens")
        self.vllm_config = vllm_config
        self.device = device
        self.num_speculative_steps = spec.num_speculative_tokens
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dtype = vllm_config.model_config.dtype
        draft_config = spec.draft_model_config
        self.hidden_size = draft_config.get_hidden_size()
        self.vocab_size = draft_config.get_vocab_size()
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.hidden_states = torch.empty(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.draft_tokens = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.last_token_indices = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.idx_mapping = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self.draft_attn_layer_names: set[str] = set()
        self.block_tables = None

    def load_model(self, target_model) -> None:
        # vLLM's loader is used only to construct the model and share the target
        # embedding/head. No vLLM speculative executor is retained.
        target_attn_layers = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )
        with set_current_vllm_config(self.vllm_config):
            self.model = load_eagle_model(target_model, self.vllm_config)
        all_attn_layers = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )
        self.draft_attn_layer_names = all_attn_layers - target_attn_layers

    def set_attn(
        self,
        _model_state,
        kv_cache_config,
        block_tables,
        _target_input_buffers,
        _target_attn_groups,
    ):
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.attn_groups, _, _ = init_attn_backend(
            kv_cache_config,
            self.vllm_config,
            self.device,
            active_layer_names=self.draft_attn_layer_names,
        )

    def _draft_attn_metadata(self, input_batch, num_reqs, num_tokens):
        query_start_cpu = input_batch.query_start_loc_np[: num_reqs + 1]
        query_start_gpu = self.input_buffers.query_start_loc[: num_reqs + 1]
        seq_lens_cpu = input_batch.seq_lens_cpu_upper_bound[:num_reqs].clone()
        seq_lens_cpu.clamp_(max=self.max_model_len)
        blocks = [x[:num_reqs] for x in self.block_tables.input_block_tables]
        slots = self.block_tables.compute_slot_mappings(
            self.idx_mapping[:num_reqs],
            query_start_gpu,
            self.input_buffers.positions[:num_tokens],
            num_tokens,
        )
        metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_gpu,
            query_start_loc_cpu=torch.from_numpy(query_start_cpu),
            max_query_len=int(input_batch.num_scheduled_tokens[:num_reqs].max()),
            seq_lens=self.input_buffers.seq_lens[:num_reqs],
            max_seq_len=self.max_model_len,
            block_tables=blocks,
            slot_mappings=slots,
            kv_cache_config=self.kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
        )
        return metadata, build_slot_mappings_by_layer(slots, self.kv_cache_config)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Keep draft execution eager; the worker owns target graph replay."""
        del cudagraph_mode

    def capture(self) -> None:
        # Draft execution remains eager until its fixed input/metadata buffers
        # have been populated by the first target step. Target decode graphs are
        # captured by LightDeltaGraphManager; no vLLM speculator graph is used.
        return

    def _draft_forward(self, input_ids, positions, hidden, attn_metadata, slots, num_tokens, step):
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=slots,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        ):
            return self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden,
                spec_step_idx=step,
            )

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ) -> torch.Tensor:
        if input_batch.num_reqs == 0:
            return self.draft_tokens[:0]
        warmup = all(req_id.startswith("_warmup_") for req_id in input_batch.req_ids)
        if not warmup and not torch.all(temperature[: input_batch.num_reqs] == 0):
            raise ValueError("LightDelta MTP currently requires greedy temperature=0")
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        target_input_ids = input_batch.input_ids[:num_tokens]
        target_positions = input_batch.positions[:num_tokens]
        starts = input_batch.query_start_loc[: num_reqs + 1].to(torch.int64)
        lengths = torch.as_tensor(
            input_batch.num_scheduled_tokens[:num_reqs],
            dtype=torch.int64,
            device=self.device,
        )
        rejected = num_rejected[:num_reqs].to(torch.int64)
        true_lengths = lengths - rejected
        last = starts[:num_reqs] + true_lengths - 1
        self.last_token_indices[:num_reqs].copy_(last)
        self.idx_mapping[:num_reqs].copy_(input_batch.idx_mapping)
        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(
            input_batch.query_start_loc[: num_reqs + 1]
        )
        self.input_buffers.seq_lens[:num_reqs].copy_(input_batch.seq_lens[:num_reqs])

        # Shift the target sequence left by one token and append the sampled
        # token. This is the MTP input contract for the Qwen3.5 MTP layer.
        self.input_buffers.input_ids[:num_tokens].copy_(target_input_ids)
        for row in range(num_reqs):
            begin = int(starts[row])
            end = int(starts[row + 1])
            keep = int(true_lengths[row])
            if keep > 1:
                self.input_buffers.input_ids[begin : begin + keep - 1].copy_(
                    target_input_ids[begin + 1 : begin + keep]
                )
            replacement = (
                last_sampled[self.idx_mapping[row], 0]
                if num_sampled[row]
                else next_prefill_tokens[0, self.idx_mapping[row]]
            )
            self.input_buffers.input_ids[begin + keep - 1] = replacement
            self.last_token_indices[row] = begin + keep - 1
            self.input_buffers.positions[begin:end].copy_(target_positions[begin:end])
        self.hidden_states[:num_tokens].copy_(last_hidden_states[:num_tokens])

        if self.block_tables is not None:
            draft_metadata, draft_slots = self._draft_attn_metadata(
                input_batch, num_reqs, num_tokens
            )
        else:
            draft_metadata, draft_slots = attn_metadata, slot_mappings
        draft_output = self._draft_forward(
            self.input_buffers.input_ids[:num_tokens],
            self.input_buffers.positions[:num_tokens],
            self.hidden_states[:num_tokens],
            draft_metadata,
            draft_slots,
            num_tokens,
            0,
        )
        if isinstance(draft_output, tuple):
            draft_hidden = draft_output[0]
        else:
            draft_hidden = draft_output
        logits = self.model.compute_logits(draft_hidden[self.last_token_indices[:num_reqs]])
        self.draft_tokens[:num_reqs, 0] = logits.argmax(dim=-1)

        # Each following draft step consumes the previous proposal. Metadata and
        # cache ownership are supplied by the vLLM scheduler; the loop and state
        # transition are LightDelta-owned.
        for step in range(1, self.num_speculative_steps):
            self.hidden_states[:num_tokens].copy_(draft_hidden[:num_tokens])
            for row in range(num_reqs):
                begin = int(starts[row])
                end = int(self.last_token_indices[row])
                if end > begin:
                    self.input_buffers.input_ids[begin:end].copy_(
                        self.input_buffers.input_ids[begin + 1 : end + 1].clone()
                    )
                self.input_buffers.input_ids[end] = self.draft_tokens[row, step - 1]
            out = self._draft_forward(
                self.input_buffers.input_ids[:num_tokens],
                self.input_buffers.positions[:num_tokens],
                self.hidden_states[:num_tokens],
                draft_metadata,
                draft_slots,
                num_tokens,
                step,
            )
            draft_hidden = out[0] if isinstance(out, tuple) else out
            self.draft_tokens[:num_reqs, step] = self.model.compute_logits(
                draft_hidden[self.last_token_indices[:num_reqs]]
            ).argmax(dim=-1)
        return self.draft_tokens[:num_reqs]


class LightDeltaRejectionSampler:
    """Greedy prefix verification for the fixed MTP service configuration."""

    def __init__(self, num_speculative_steps: int):
        self.num_speculative_steps = num_speculative_steps

    @torch.inference_mode()
    def __call__(self, logits, input_batch, draft_logits=None):
        if draft_logits is not None:
            raise ValueError("LightDelta greedy sampler does not accept draft logits")
        num_reqs = input_batch.num_reqs
        target = logits.argmax(dim=-1)
        offsets = input_batch.cu_num_logits[: num_reqs + 1].tolist()
        input_offsets = input_batch.query_start_loc[: num_reqs + 1].tolist()
        token_dtype = input_batch.input_ids.dtype
        token_device = input_batch.input_ids.device
        sampled = torch.full(
            (num_reqs, self.num_speculative_steps + 1),
            -1,
            dtype=token_dtype,
            device=token_device,
        )
        num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=token_device)
        num_rejected = torch.zeros_like(num_sampled)
        for row in range(num_reqs):
            begin, end = offsets[row], offsets[row + 1]
            draft_count = int(input_batch.num_draft_tokens_per_req[row])
            if draft_count == 0:
                sampled[row, 0] = target[begin]
                if int(input_batch.seq_lens[row]) < int(input_batch.prefill_len_np[row]):
                    num_sampled[row] = 0
                    num_rejected[row] = 0
                else:
                    num_sampled[row] = 1
                continue
            if draft_count > self.num_speculative_steps:
                raise ValueError("invalid MTP draft-token metadata")
            input_end = input_offsets[row + 1]
            draft = input_batch.input_ids[input_end - draft_count : input_end]
            accepted = 0
            while accepted < draft_count:
                if begin + accepted >= end or target[begin + accepted] != draft[accepted]:
                    break
                sampled[row, accepted] = draft[accepted]
                accepted += 1
            bonus_index = min(begin + accepted, end - 1)
            sampled[row, accepted] = target[bonus_index]
            if int(input_batch.seq_lens[row]) < int(input_batch.prefill_len_np[row]):
                # Chunked prefills carry draft slots for the following MTP
                # step, but do not emit output tokens or roll back cache
                # positions. This matches vLLM's rejection-sampler contract.
                num_sampled[row] = 0
                num_rejected[row] = 0
            else:
                num_sampled[row] = accepted + 1
                # Roll back the target positions that were verified but did
                # not become output tokens, using the actual logits span.
                num_rejected[row] = (end - begin) - (accepted + 1)
        return SamplerOutput(sampled, None, None, num_sampled, num_rejected)
