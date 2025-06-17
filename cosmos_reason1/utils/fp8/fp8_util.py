# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from strenum import StrEnum
from abc import abstractmethod, ABC
from typing import Union, List, Dict, Type
from functools import partial
from torchao.float8 import convert_to_float8_training
from torchao.float8 import Float8LinearConfig
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

from cosmos_reason1.policy.config import Config as CosmosConfig
from cosmos_reason1.utils.parallelism import ParallelDims
from cosmos_reason1.utils.util import is_cuda_compatible, torch_version_at_least
from cosmos_reason1.utils.logging import logger

MIN_TORCH_VERSION_FOR_FP8 = "2.7.0"
IS_TORCH_COMPATIBLE_WITH_FP8 = torch_version_at_least(MIN_TORCH_VERSION_FOR_FP8)

if not IS_TORCH_COMPATIBLE_WITH_FP8:
    logger.warning(
        f"[FP8] FP8 is not supported for this version of PyTorch, minimum version required: {MIN_TORCH_VERSION_FOR_FP8}, but got: {torch.__version__}. FP8 setting will take no effect."
    )

# Mainly refer to: https://github.com/pytorch/torchtitan/blob/main/docs/float8.md


class ModelConverter(ABC):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims):
        self.config = config
        self.parallel_dims = parallel_dims

    @abstractmethod
    def convert_model(self, model: torch.nn.Module) -> torch.nn.Module: ...

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        """
        Post-optimizer hook (e.g. compute weights statistics).
        """
        ...


_COSMOS_MODEL_CONVERTER_REGISTRY: Dict[str, Type[ModelConverter]] = {}


def register_class(reg_key: str, *, allow_override: bool = False):
    def decorator(cls: Type) -> Type:
        if not allow_override and reg_key in _COSMOS_MODEL_CONVERTER_REGISTRY:
            raise ValueError(f"Class '{reg_key}' is already registered.")
        _COSMOS_MODEL_CONVERTER_REGISTRY[reg_key] = cls
        return cls

    return decorator


class FP8Recipe(StrEnum):
    DYNAMIC_SCALING = "dynamic_scaling"
    DELAYED_SCALING = "delayed_scaling"


def is_valid_fp8_recipe(value: str) -> bool:
    return value in FP8Recipe.__members__.values()


class FP8QuantRecipe(StrEnum):
    ROWWISE = "rowwise"
    TENSORWISE = "tensorwise"


def is_valid_fp8_quant_recipe(value: str) -> bool:
    return value in FP8QuantRecipe.__members__.values()


# Refer to: https://github.com/pytorch/torchtitan/blob/e7c0cae934df78d6e9c2835f42ff1f757dc3fddc/torchtitan/components/quantization/utils.py#L10
def module_filter_fn(mod: nn.Module, fqn: str, filter_fqns: list[str]) -> bool:
    """
    Filter function to determine which modules should be converted.
    For both Float8 and MXFP8, we only convert Linear modules
    with dimensions divisible by 16 and not matching any filtered FQNs.

    Args:
        mod: The module to be converted.
        fqn: The fully qualified name of the module.
        filter_fqns: The list of FQNs to filter.

    Returns:
        True if the module should be converted, False otherwise.
    """
    if not isinstance(mod, nn.Linear):
        return False

    # All dims must be divisible by 16 due to float8 tensorcore hardware requirements.
    dims_multiples_of_16 = (
        mod.weight.shape[0] % 16 == 0 and mod.weight.shape[1] % 16 == 0
    )

    # If the fqn matches any filtered fqn, then we should not convert this module.
    is_filtered_fqn = any(filter_fqn in fqn for filter_fqn in filter_fqns)

    return dims_multiples_of_16 and not is_filtered_fqn


@register_class("fp8")
class FP8ModelConverter(ModelConverter):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims):
        super().__init__(config, parallel_dims)
        if not IS_TORCH_COMPATIBLE_WITH_FP8:
            return

        if not is_cuda_compatible(8, 9):
            raise RuntimeError(
                "FP8 is only supported for device that has compute capability 8.9 or higher"
            )
        self.fp8_config = config.train.fp8

        assert is_valid_fp8_quant_recipe(self.fp8_config.quant_recipe)
        assert is_valid_fp8_recipe(self.fp8_config.fp8_recipe)

        if self.fp8_config.fp8_recipe == FP8Recipe.DELAYED_SCALING:
            raise NotImplementedError("[FP8] Delayed scaling is not supported yet.")

        self.precompute_scale = False

        if self.fp8_config.quant_recipe == "rowwise":
            # From torchtitan, it reports an issue that RMSNorm will cause NaN when rowwise quantization and torch.compile is enabled,
            # From that issue, it is recommended to set torch._inductor.config.emulate_precision_casts to True to avoid this.
            # Issue: https://github.com/pytorch/pytorch/issues/150859
            torch._inductor.config.emulate_precision_casts = True
            self.ao_float8_config = Float8LinearConfig.from_recipe_name(
                self.fp8_config.quant_recipe
            )
            logger.debug(
                "[FP8] Set torch._inductor.config.emulate_precision_casts to True"
            )
        elif self.fp8_config.quant_recipe == "tensorwise":
            # For tensorwise, torchao supports that precompute scale and perform FSDP2 weight all-gather in FP8.
            # this could save the bandwidth.
            # self.precompute_scale = True
            # TODO: (lms) WeightWithDynamicFloat8CastTensor is used in torchao, this could cause
            # the weight sync got nullptr (WeightWithDynamicFloat8CastTensor is a wrapper).
            #  Just disable the enable_fsdp_float8_all_gather and precompute_scale for now. Handle it later.
            # enable_fsdp_float8_all_gather = (
            #     parallel_dims.dp_shard_enabled
            # )  # Unlike torchtitan, we enable it by default if in FSDP2.
            self.precompute_scale = False
            enable_fsdp_float8_all_gather = False
            self.ao_float8_config = Float8LinearConfig(
                enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
                # It is recommended to set force_recompute_fp8_weight_in_bwd to True for FSDP2.
                force_recompute_fp8_weight_in_bwd=True,
            )

    def convert_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if not IS_TORCH_COMPATIBLE_WITH_FP8:
            return

        if not self.fp8_config.enable_fp8:
            return

        convert_to_float8_training(
            model,
            config=self.ao_float8_config,
            module_filter_fn=partial(
                module_filter_fn, filter_fqns=model.fqn_filter_for_fp8()
            ),
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        if not IS_TORCH_COMPATIBLE_WITH_FP8:
            return

        if not self.fp8_config.enable_fp8:
            return

        if not self.precompute_scale:
            return

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_float8_dynamic_scale_for_fsdp(m)
