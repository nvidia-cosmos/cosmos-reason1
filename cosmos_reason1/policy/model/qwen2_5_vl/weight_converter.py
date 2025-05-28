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

from cosmos_reason1.policy.model.gpt.weight_converter import (
    convert_weight_from_hf as gpt_weight_from_hf,
)
from cosmos_reason1.utils.parallelism import ParallelDims
import torch
import re
from typing import Tuple, Dict, Any
from cosmos_reason1.utils.parallelism_registry import (
    register_rollout_parallelism_strategy,
    register_policy_parallelism_strategy,
)


def map_key_from_hf(name: str, src_model_type: str) -> str:
    if src_model_type in ["llama", "qwen2", "qwen3", "qwen2_5_vl"]:
        return name.replace("model.", "")
    else:
        raise ValueError(f"Unsupported model type: {src_model_type}")


def convert_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    src_model_type: str,
    parallel_dims: ParallelDims,
    ignore_unknown_weights: bool = False,
) -> Tuple[str, torch.Tensor]:
    lm_part_name, lm_part_shard = gpt_weight_from_hf(
        tensor, name, src_model_type, parallel_dims, ignore_unknown_weights=True
    )
    if lm_part_name is not None:
        return lm_part_name, lm_part_shard
    assert name.startswith("visual."), f"Unsupported weight: {name}"

    if (
        parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
        or parallel_dims.tp_enabled
    ):
        dp_shard_rank = parallel_dims.mesh["dp_cp_tp"].get_local_rank()
        dp_shard_size = parallel_dims.mesh["dp_cp_tp"].size()
    else:
        dp_shard_rank = 0
        dp_shard_size = 1

    dest_name = name.replace("visual.", "")
    shard = tensor
    # TODO(cjx): Only FSDP sharding is supported for visual part
    shard = shard.contiguous()
    shard = shard.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
    return dest_name, shard.contiguous()


@register_rollout_parallelism_strategy("qwen2_5_vl")
def map_weight_parallel_dims(
    shape: Tuple[int], dest_name: str, parallel_dims: ParallelDims, model_config: Any
) -> Tuple[Dict[str, int], Dict[int, list], int]:
    tp_size = parallel_dims.tp
    dp_shard_size = parallel_dims.dp_shard * parallel_dims.cp

    dims_map = {}
    dim = "tp"
    no_dp = False
    assert dest_name.startswith("visual.")
    if tp_size > 1:
        if (
            match := re.search(r"patch_embed\.proj\.(weight|bias)", dest_name)  # noqa: F841
        ) is not None:
            pass
        elif (
            match := re.search(r"merger\.ln_q\.(weight|bias)", dest_name)  # noqa: F841
        ) is not None:
            pass
        elif (
            match := re.search(r"merger\.mlp\.0\.(weight|bias)", dest_name)  # noqa: F841
        ) is not None:
            dims_map[dim] = 0
        elif (match := re.search(r"merger\.mlp\.2\.weight", dest_name)) is not None:  # noqa: F841
            dims_map[dim] = len(shape) - 1
        elif (match := re.search(r"merger\.mlp\.2\.bias", dest_name)) is not None:  # noqa: F841
            pass
        elif (
            match := re.search(  # noqa: F841
                r"blocks\.(\d+)\.attn\.(q|k|v)\.(weight|bias)",
                dest_name,
            )
        ) is not None:
            dims_map[dim] = 0
            no_dp = True
        elif (
            match := re.search(r"blocks\.(\d+)\.attn\.proj\.weight", dest_name)  # noqa: F841
        ) is not None:
            dims_map[dim] = len(shape) - 1
        elif (
            match := re.search(r"blocks\.(\d+)\.attn\.proj\.bias", dest_name)  # noqa: F841
        ) is not None:
            pass
        elif (
            match := re.search(  # noqa: F841
                r"blocks\.(\d+)\.mlp\.(up_proj|gate_proj)\.(weight|bias)", dest_name
            )
        ) is not None:
            dims_map[dim] = 0
        elif (
            match := re.search(r"blocks\.(\d+)\.mlp\.down_proj\.weight", dest_name)  # noqa: F841
        ) is not None:
            dims_map[dim] = len(shape) - 1
        elif (
            match := re.search(r"blocks\.(\d+)\.mlp\.down_proj\.bias", dest_name)  # noqa: F841
        ) is not None:
            pass
        elif (
            match := re.search(  # noqa: F841
                r"blocks\.(\d+)\.(norm1|norm2)\.(weight|bias)", dest_name
            )
        ) is not None:
            pass
        else:
            raise ValueError(f"Unsupported weight: {dest_name}")
    else:
        pass

    # Do FSDP sharding
    dim = "dp_shard_cp"
    if dp_shard_size > 1 and not no_dp:
        dims_map[dim] = 0
    else:
        pass

    tensor_dim_to_parallel_map = {}
    for k, v in dims_map.items():
        if v not in tensor_dim_to_parallel_map:
            tensor_dim_to_parallel_map[v] = []
        tensor_dim_to_parallel_map[v].append(k)
    pp_rank = 0
    return dims_map, tensor_dim_to_parallel_map, pp_rank


@register_policy_parallelism_strategy("qwen2_5_vl")
def map_weight_parallel_dims_no_visual_tp(
    shape: Tuple[int], dest_name: str, parallel_dims: ParallelDims, model_config: Any
) -> Tuple[Dict[str, int], Dict[int, list], int]:
    tp_size = parallel_dims.tp
    dp_shard_size = parallel_dims.dp_shard * parallel_dims.cp

    dims_map = {}
    no_dp = False
    assert tp_size == 1, f"tp_size should be 1, but got {tp_size}"
    if dest_name.startswith("visual."):
        assert (
            dest_name.startswith("visual.patch_embed.")
            or dest_name.startswith("visual.merger.")
            or dest_name.startswith("visual.blocks.")
        )
        if (
            match := re.search(  # noqa: F841
                r"blocks\.(\d+)\.attn\.(q|k|v)\.(weight|bias)",
                dest_name,
            )
        ) is not None:
            no_dp = True

        dim = "dp_shard_cp"
        if dp_shard_size > 1 and not no_dp:
            dims_map[dim] = 0
        else:
            pass

        tensor_dim_to_parallel_map = {}
        for k, v in dims_map.items():
            if v not in tensor_dim_to_parallel_map:
                tensor_dim_to_parallel_map[v] = []
            tensor_dim_to_parallel_map[v].append(k)
        pp_rank = 0
        return dims_map, tensor_dim_to_parallel_map, pp_rank
    else:
        raise ValueError(f"Unsupported weight for policy parallelism: {dest_name}")
