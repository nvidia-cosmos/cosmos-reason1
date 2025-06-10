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

from typing import Dict, Type, Tuple, List
import torch
from torch import nn
from cosmos_reason1.utils.parallelism import ParallelismConfig
from cosmos_reason1.utils.parallelism_map import (
    slice_tensor_with_strategies,
    DimRankInfo,
)
from abc import ABC
from cosmos_reason1.utils.util import seperate_nccl_comm_needed
from cosmos_reason1.utils.pynccl import nccl_recv


class WeightMapper(ABC):
    def __init__(self, hf_config_path: str):
        self.compatible_weight_map: Dict[str, torch.Tensor] = None
        self.compatible_list: List[Tuple[str, Tuple[int]]]

    def name_p2r(self, policy_weight_name: str) -> str:
        pass

    def name_to_hf(self, rollout_weight_name: str) -> str:
        pass

    def split_qkv_weight(self, name, weight: torch.Tensor):
        pass

    def split_gate_proj_weight(self, name, weight: torch.Tensor):
        pass

    def generate_compatible_map(self, model: nn.Module):
        self.compatible_weight_map, self.compatible_list = (
            self.custom_generate_compatible_map(model)
        )

        return self.compatible_weight_map, self.compatible_list

    def name_to_model_index(self, dest_name: str) -> int:
        pass

    def get_rollout_parallelism(self, parallelism_config: ParallelismConfig):
        pass

    def get_policy_parallelism(self, parallelism_config: ParallelismConfig):
        pass

    def get_policy_parallelism_strategy(self):
        pass

    def get_rollout_parallelism_strategy(self):
        pass

    def recv_weight_shard(
        self,
        global_rank_of_rollout: int,
        inst: Tuple[int, int, Dict[int, DimRankInfo], str, Tuple[int]],
        communicator_index: Dict[int, int],
        do_weight_sync_check: bool = False,
    ):
        need_sep_comm = seperate_nccl_comm_needed()

        p_rank, r_rank, tensor_split_strategys, dest_name, shape = inst
        assert r_rank == global_rank_of_rollout

        target_tensor = self.compatible_weight_map[dest_name]

        view = slice_tensor_with_strategies(target_tensor, tensor_split_strategys)

        if do_weight_sync_check:
            cloned_target_tensor = target_tensor.clone()
            # clear the current view
            view.zero_()

        recv_tensor = None
        if view.is_contiguous():
            recv_tensor = view
        else:
            # new a temp tensor
            recv_tensor = torch.empty_like(view)

        # logger.info(
        #     f"[Rollout] rank {global_rank_of_rollout} recv tensor: {dest_name} from rank {p_rank} with shape: {view.shape} out of {target_tensor.shape} with dtype {view.dtype} on device {view.device}"
        # )
        nccl_recv(
            recv_tensor, 0 if need_sep_comm else p_rank, communicator_index[p_rank]
        )

        # inplace copy
        if not view.is_contiguous():
            view.copy_(recv_tensor)

        if do_weight_sync_check:
            # If the weight sync between Policy and Rollout is correct, the
            # `target_tensor` would have no change.
            # TODO: (lms) When we support quantization in rollout side,
            # we should handle the numerical error of quantized weight, not
            # just apply `torch.allclose` simply.
            if not torch.allclose(cloned_target_tensor, target_tensor):
                raise ValueError(
                    f"Weight sync check failed after weight sync instruction: {inst}"
                )

        return recv_tensor.numel() * recv_tensor.element_size()


_MODEL_WEIGHT_MAPPER_REGISTRY: Dict[str, Type[WeightMapper]] = {}


def register_class(reg_key: str, *, allow_override: bool = False):
    def decorator(cls: Type) -> Type:
        if not allow_override and reg_key in _MODEL_WEIGHT_MAPPER_REGISTRY:
            raise ValueError(f"Class '{reg_key}' is already registered.")
        _MODEL_WEIGHT_MAPPER_REGISTRY[reg_key] = cls
        return cls

    return decorator


def get_weight_mapper(model_type: str) -> Type[WeightMapper]:
    if model_type not in _MODEL_WEIGHT_MAPPER_REGISTRY:
        raise ValueError(f"ModelType '{model_type}' is not supported now.")

    return _MODEL_WEIGHT_MAPPER_REGISTRY[model_type]
