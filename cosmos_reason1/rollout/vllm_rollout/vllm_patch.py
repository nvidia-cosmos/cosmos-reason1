import time
import types

import torch
import torch.nn as nn

import vllm
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

if vllm.__version__ >= "0.9.0":
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
else:
    from vllm.model_executor.model_loader.loader import (
        _process_weights_after_loading as process_weights_after_loading,
    )
    from vllm.model_executor.model_loader.loader import DefaultModelLoader


from vllm.utils import GiB_bytes
from vllm.utils import DeviceMemoryProfiler
from vllm.logger import init_logger
from vllm import LLM


logger = init_logger(__name__)


"""
This patch is for supporting weight reloading in vllm.
It can only be used for V1 vLLM backend.

"""


class CustomModelLoader(DefaultModelLoader):
    def reload_weights(self, vllm_config: VllmConfig, model: nn.Module) -> None:
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                self.get_all_weights(model_config, model)
            )
            self.counter_after_loading_weights = time.perf_counter()
            logger.info(
                "Loading weights took %.2f seconds",
                self.counter_after_loading_weights
                - self.counter_before_loading_weights,
            )
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            if model_config.quantization is None and loaded_weights is not None:
                weights_not_loaded = weights_to_load - loaded_weights
                if weights_not_loaded:
                    raise ValueError(
                        "Following weights were not initialized from "
                        f"checkpoint: {weights_not_loaded}"
                    )

            process_weights_after_loading(model, model_config, target_device)


def patch_vllm_model_to_reload_weight(llm: LLM):
    def add_method(obj, method, method_name):
        setattr(obj, method_name, types.MethodType(method, obj))

    def reload_model_worker(self) -> None:
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be " "used for one instance per process."
            )
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext

            context = nullcontext()
        with context:
            self.model_runner.reload_model()

    add_method(
        llm.llm_engine.engine_core.engine_core.model_executor.driver_worker,
        reload_model_worker,
        "reload_model",
    )

    def reload_model_runner(self) -> None:
        logger.info("Starting to realod weight for current vllm model")
        with DeviceMemoryProfiler(self.device) as m:
            time_before_load = time.perf_counter()
            loader = CustomModelLoader(self.vllm_config.load_config)
            if not hasattr(loader, "reload_weights"):
                raise ValueError("Model loader does not support reloading weights")
            loader.reload_weights(self.vllm_config, self.model)
            time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model weight reloading took %.4f GiB and %.6f seconds",
            self.model_memory_usage / GiB_bytes,
            time_after_load - time_before_load,
        )

    add_method(
        llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner,
        reload_model_runner,
        "reload_model",
    )
