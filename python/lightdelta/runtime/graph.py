"""Fixed-shape CUDA Graph runtime owned by LightDelta.

The service still uses vLLM for model construction, scheduler metadata and cache
allocation. Capture, descriptor selection and replay live here so the optimization
is explicit and does not silently select vLLM's graph manager.
"""

from dataclasses import dataclass

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger("lightdelta.graph")


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """Shape descriptor exchanged with vLLM's scheduler dispatch helper."""

    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None
    uniform_token_count: int | None = None
    max_query_len: int | None = None
    num_active_loras: int = 0


@dataclass
class _Captured:
    descriptor: BatchExecutionDescriptor
    graph: torch.cuda.CUDAGraph
    hidden: torch.Tensor
    aux_hidden: list[torch.Tensor]


def _prepare_capture_inputs(
    num_reqs,
    num_tokens,
    model_state,
    input_buffers,
    block_tables,
    attn_groups,
    kv_cache_config,
    query_len,
):
    """Build capturable attention metadata without vLLM's graph manager."""
    input_batch = InputBatch.make_dummy(
        num_reqs, num_tokens, input_buffers, max_query_len=query_len
    )
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(slot_mappings, kv_cache_config)
    attn_metadata = model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=True,
    )
    return attn_metadata, slot_mappings_by_layer


class LightDeltaGraphManager:
    """Own full decode graph capture and replay for one fixed service process.

    This intentionally supports the project's single-GPU, no-LoRA, full-decode
    configuration. Unsupported graph modes are rejected at construction instead
    of falling back to another graph implementation.
    """

    def __init__(self, vllm_config, device: torch.device, decode_query_len: int):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.decode_query_len = decode_query_len
        self.capture_sizes = tuple(
            sorted(vllm_config.compilation_config.cudagraph_capture_sizes or ())
        )
        self.enabled = bool(self.capture_sizes)
        if self.enabled and vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("LightDelta graph runtime requires tensor_parallel_size=1")
        self.graphs: dict[BatchExecutionDescriptor, _Captured] = {}
        self._capture_descs = {
            CUDAGraphMode.FULL: [self._descriptor(size) for size in self.capture_sizes]
        }
        self._graphs_captured = False
        self.pool = None
        self.use_breakable_cg = False
        self._max_full_descs_to_capture = None
        self._capture_mem_samples = None

    def _descriptor(self, num_tokens: int) -> BatchExecutionDescriptor:
        if num_tokens % self.decode_query_len:
            raise ValueError("capture sizes must be divisible by decode query length")
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=num_tokens,
            num_reqs=num_tokens // self.decode_query_len,
            uniform_token_count=self.decode_query_len,
        )

    def needs_capture(self) -> bool:
        return self.enabled and bool(self.capture_sizes)

    def init_breakable_cg_runner(self, model) -> None:
        raise RuntimeError("LightDelta uses full graphs and has no breakable graph mode")

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int = 0,
        max_query_len: int | None = None,
    ) -> BatchExecutionDescriptor:
        if num_active_loras or not self.enabled or uniform_token_count != self.decode_query_len:
            return BatchExecutionDescriptor(CUDAGraphMode.NONE, num_tokens, num_reqs)
        for descriptor in self._capture_descs[CUDAGraphMode.FULL]:
            if descriptor.num_tokens >= num_tokens and descriptor.num_reqs >= num_reqs:
                return descriptor
        return BatchExecutionDescriptor(CUDAGraphMode.NONE, num_tokens, num_reqs)

    @torch.inference_mode()
    def capture(
        self,
        model,
        model_state,
        input_buffers,
        intermediate_tensors,
        block_tables,
        attn_groups,
        kv_cache_config,
        pcp_manager=None,
        has_lora=False,
        use_aux_hidden_state_outputs=False,
        lora_capture_hook=None,
        progress_bar_desc="Capturing LightDelta CUDA graphs",
    ) -> None:
        if not self.enabled or self._graphs_captured:
            return
        logger.warning(
            "Capturing LightDelta decode graphs: sizes=%s query_len=%d",
            self.capture_sizes,
            self.decode_query_len,
        )
        if has_lora or use_aux_hidden_state_outputs or pcp_manager is not None:
            raise ValueError("LightDelta graph runtime supports the fixed single-GPU path only")
        if lora_capture_hook is not None:
            raise ValueError("LightDelta graph runtime does not capture LoRA variants")

        input_buffers.is_padding.fill_(True)
        for descriptor in self._capture_descs[CUDAGraphMode.FULL]:
            num_tokens = descriptor.num_tokens
            num_reqs = descriptor.num_reqs
            attn_metadata, slot_mappings = _prepare_capture_inputs(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                self.decode_query_len,
            )
            model_inputs = {
                "input_ids": input_buffers.input_ids[:num_tokens],
                "positions": input_buffers.positions[:num_tokens],
                **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
            }
            hidden_box: list[torch.Tensor | None] = [None]
            aux_box: list[list[torch.Tensor]] = [[]]

            def forward(
                attn_metadata=attn_metadata,
                slot_mappings=slot_mappings,
                num_tokens=num_tokens,
                model_inputs=model_inputs,
                hidden_box=hidden_box,
                aux_box=aux_box,
            ) -> None:
                with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=slot_mappings,
                    batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
                    is_padding=input_buffers.is_padding[:num_tokens],
                ):
                    output = model(**model_inputs)
                if isinstance(output, tuple):
                    hidden, aux = output
                else:
                    hidden, aux = output, []
                if hidden_box[0] is None:
                    hidden_box[0] = torch.empty_like(hidden)
                    aux_box[0] = [torch.empty_like(x) for x in aux]
                hidden_box[0].copy_(hidden)
                for dst, src in zip(aux_box[0], aux, strict=True):
                    dst.copy_(src)

            forward()
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            assert hidden_box[0] is not None
            self.graphs[descriptor] = _Captured(descriptor, graph, hidden_box[0], aux_box[0])
        input_buffers.is_padding.fill_(False)
        self._graphs_captured = True
        logger.warning("LightDelta decode graphs captured: sizes=%s", self.captured_token_counts())

    def run_fullgraph(self, descriptor: BatchExecutionDescriptor):
        captured = self.graphs.get(descriptor)
        if captured is None:
            raise RuntimeError(f"LightDelta graph was not captured: {descriptor}")
        captured.graph.replay()
        if captured.aux_hidden:
            return captured.hidden[: descriptor.num_tokens], [
                x[: descriptor.num_tokens] for x in captured.aux_hidden
            ]
        return captured.hidden[: descriptor.num_tokens]

    def run_pw_graph(self, model, model_inputs):
        raise RuntimeError("LightDelta does not use vLLM piecewise graphs")

    def captured_token_counts(self) -> list[int]:
        return sorted({d.num_tokens for d in self.graphs})
