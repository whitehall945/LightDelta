"""vLLM worker shim that replaces only graph and speculative execution."""

import os

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker

from lightdelta.config import BackendConfig

from .graph import LightDeltaGraphManager
from .speculation import LightDeltaRejectionSampler, LightDeltaSpeculator

logger = init_logger("lightdelta.worker")


class LightDeltaWorker(Worker):
    """Keep vLLM loading, scheduling and cache allocation; own D/E execution."""

    def _config(self) -> BackendConfig:
        return BackendConfig.from_file(os.environ["LIGHTDELTA_CONFIG"])

    def init_device(self):
        super().init_device()
        config = self._config()
        runner = self.model_runner
        runner.cudagraph_manager = (
            LightDeltaGraphManager(
                self.vllm_config,
                self.device,
                config.num_speculative_tokens + 1,
            )
            if config.cuda_graph
            else LightDeltaGraphManager(self.vllm_config, self.device, 1)
        )
        # vLLM's memory estimator assumes its own manager internals. Graph
        # capture is already owned by LightDelta, so reserve no extra estimate
        # and let the real capture use the measured allocator state.
        runner.profile_cudagraph_memory = lambda: 0
        # Prevent vLLM from constructing or executing its own MTP module. The
        # model loader is still invoked by this worker after target construction.
        if config.speculation_method == "mtp":
            runner.speculator = None
            runner.rejection_sampler = None

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        config = self._config()
        if config.speculation_method != "mtp":
            return
        runner = self.model_runner
        speculator = LightDeltaSpeculator(self.vllm_config, self.device)
        speculator.load_model(runner.model)
        runner.speculator = speculator
        runner.rejection_sampler = LightDeltaRejectionSampler(config.num_speculative_tokens)

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_from_config(kv_cache_config)
        config = self._config()
        runner = self.model_runner
        if config.speculation_method == "mtp":
            runner.speculator.set_attn(
                runner.model_state,
                runner.kv_cache_config,
                runner.block_tables,
                runner.input_buffers,
                runner.attn_groups,
            )
        runner.cudagraph_manager = LightDeltaGraphManager(
            self.vllm_config,
            self.device,
            runner.decode_query_len,
        )
        runner.profile_cudagraph_memory = lambda: 0
        logger.warning(
            "LightDelta runtime installed after KV cache initialization: graph=%s, mtp=%s",
            config.cuda_graph,
            config.speculation_method == "mtp",
        )
