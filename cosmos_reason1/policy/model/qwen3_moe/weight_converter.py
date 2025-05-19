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

from cosmos_reason1.utils.parallelism import ParallelDims
import torch
import re
from typing import Tuple, Dict, Any
from cosmos_reason1.utils.parallelism_registry import register_parallelism_strategy


def map_key_from_hf(name: str, src_model_type: str) -> str:
    return name.replace("model.", "")


def convert_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    src_model_type: str,
    parallel_dims: ParallelDims,
    n_experts: int,
    ignore_unknown_weights: bool = False,
) -> Tuple[str, torch.Tensor]:
    tp_ep_rank, tp_ep_size = parallel_dims.tp_coord
    assert n_experts % tp_ep_size == 0, "n_experts must be divisible by tp_ep_size"

    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        dp_shard_rank = parallel_dims.mesh[tuple(("dp_shard_cp",))].get_local_rank()
        dp_shard_size = parallel_dims.mesh[tuple(("dp_shard_cp",))].size()
    else:
        dp_shard_rank = 0
        dp_shard_size = 1

    # Expert weight are aggregated into (n_experts, in_features, out_features)
    # Weight are loaded in (out_features, in_features) shape
    # So we do not do FSDP sharding on expert weights, instead we filter by expert id
    should_do_fsdp_sharding = True

    dest_name = map_key_from_hf(name, src_model_type)

    if "lm_head.weight" == dest_name:
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif "lm_head.bias" == dest_name:
        shard = tensor
    elif "embed_tokens.weight" == dest_name:
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif dest_name in ["norm.weight", "norm.bias"]:
        shard = tensor
    elif (
        match := re.search(
            r"layers\.(\d+)\.self_attn\.(q_norm|k_norm|v_norm)\.(weight|bias)",
            dest_name,
        )
    ) is not None:
        shard = tensor
    elif (
        match := re.search(r"layers\.(\d+)\.input_layernorm\.(weight|bias)", dest_name)
    ) is not None:
        shard = tensor
    elif (
        match := re.search(
            r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.(weight|bias)",
            dest_name,
        )
    ) is not None:
        shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
    elif (
        match := re.search(
            r"layers\.(\d+)\.self_attn\.(o_proj)\.(weight|bias)", dest_name
        )
    ) is not None:
        if dest_name.endswith(".bias"):
            shard = tensor
        else:
            shard = tensor.tensor_split(tp_ep_size, dim=-1)[tp_ep_rank]
    elif (
        match := re.search(  # noqa: F841
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(up_proj|gate_proj)\.(weight|bias)",
            dest_name,
        )
    ) is not None:
        # shard = tensor.tensor_split(tp_ep_size, dim=0)[tp_ep_rank]
        # Check whether this expert belongs to the current process
        # Groups example (with 32 experts, and 4 EP groups):
        #  EP=0: 0, 1, 2, 3, 4, 5, 6, 7
        #  EP=1: 8, 9, 10, 11, 12, 13, 14, 15
        #  EP=2: 16, 17, 18, 19, 20, 21, 22, 23
        #  EP=3: 24, 25, 26, 27, 28, 29, 30, 31
        n_expert_per_ep = n_experts // tp_ep_size
        belongs_to_current_ep = (
            tp_ep_rank * n_expert_per_ep
            <= int(match.group(2))  # Expert index
            < (tp_ep_rank + 1) * n_expert_per_ep
        )
        belongs_to_current_dp_shard = (
            int(match.group(2)) - tp_ep_rank * n_expert_per_ep
        ) // (n_expert_per_ep // dp_shard_size) == dp_shard_rank
        if belongs_to_current_ep and belongs_to_current_dp_shard:
            should_do_fsdp_sharding = False
            shard = tensor.transpose(
                -2, -1
            )  # For torch._grouped_mm, the weight should be in (in_features, out_features) shape
        else:
            # If the expert does not belong to the current process, return None to skip this weight
            return None, None
    elif (
        match := re.search(
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)", dest_name
        )  # noqa: F841
    ) is not None:
        # The same logic as the up_proj/gate_proj
        n_expert_per_ep = n_experts // tp_ep_size
        belongs_to_current_ep = (
            tp_ep_rank * n_expert_per_ep
            <= int(match.group(2))
            < (tp_ep_rank + 1) * n_expert_per_ep
        )
        belongs_to_current_dp_shard = (
            int(match.group(2)) - tp_ep_rank * n_expert_per_ep
        ) // (n_expert_per_ep // dp_shard_size) == dp_shard_rank
        if belongs_to_current_ep and belongs_to_current_dp_shard:
            should_do_fsdp_sharding = False
            shard = tensor.transpose(
                -2, -1
            )  # For torch._grouped_mm, the weight should be in (in_features, out_features) shape
        else:
            # If the expert does not belong to the current process, return None to skip this weight
            return None, None
    elif (
        match := re.search(  # noqa: F841
            r"layers\.(\d+)\.post_attention_layernorm\.(weight|bias)", dest_name
        )
    ) is not None:
        shard = tensor
    elif (
        match := re.search(  # noqa: F841
            r"layers\.(\d+)\.mlp\.gate\.weight", dest_name
        )
    ) is not None:
        # TODO(cjx): Small enough, forbid FSDP sharding is better
        shard = tensor
    elif not ignore_unknown_weights:
        raise ValueError(f"Unsupported weight: {dest_name}")
    else:
        return None, None

    # Do FSDP sharding
    shard = shard.contiguous()
    if should_do_fsdp_sharding:
        shard = shard.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
    return dest_name, shard.contiguous()


@register_parallelism_strategy("qwen3_moe")
def map_weight_parallel_dims(
    shape: Tuple[int], dest_name: str, parallel_dims: ParallelDims, model_config: Any
) -> Tuple[Dict[str, int], list, Dict[int, list]]:
    tp_ep_size = parallel_dims.tp
    dp_shard_size = parallel_dims.dp_shard * parallel_dims.cp

    dims_map = {}
    dim = "tp"

    pp_rank = 0
    pp_size = parallel_dims.pp
    n_layers = model_config.num_hidden_layers

    assert dest_name.startswith("model.") or dest_name.startswith("lm_head.")
    if tp_ep_size > 1:
        if "lm_head.weight" == dest_name:
            dims_map[dim] = 0
            pp_rank = pp_size - 1
        elif "lm_head.bias" == dest_name:
            pp_rank = pp_size - 1
            pass
        elif "model.embed_tokens.weight" == dest_name:
            dims_map[dim] = 0
            pp_rank = 0
        elif dest_name in ["model.norm.weight", "model.norm.bias"]:
            pp_rank = pp_size - 1
            pass
        else:
            if (
                match := re.search(  # noqa: F841
                    r"layers\.(\d+)\.input_layernorm\.(weight|bias)", dest_name
                )
            ) is not None:
                pass
            elif (
                match := re.search(  # noqa: F841
                    r"layers\.(\d+)\.self_attn\.(q_norm|k_norm|v_norm)\.(weight|bias)",
                    dest_name,
                )
            ) is not None:
                pass
            elif (
                match := re.search(  # noqa: F841
                    r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.(weight|bias)",
                    dest_name,
                )
            ) is not None:
                dims_map[dim] = 0
            elif (
                match := re.search(  # noqa: F841
                    r"layers\.(\d+)\.self_attn\.(o_proj)\.(weight|bias)", dest_name
                )
            ) is not None:
                dims_map[dim] = len(shape) - 1
            elif (
                match := re.search(  # noqa: F841
                    r"layers\.(\d+)\.mlp\.(up_proj|gate_proj)\.(weight|bias)", dest_name
                )
            ) is not None:
                dims_map[dim] = (
                    0  # For MoE Expert Parallelism, split in expert_num dimension
                )
            elif (
                match := re.search(  # noqa: F841
                    r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)", dest_name
                )
            ) is not None:
                dims_map[dim] = (
                    0  # For MoE Expert Parallelism, split in expert_num dimension
                )
            elif (
                match := re.search(  # noqa: F841
                    r"layers\.(\d+)\.post_attention_layernorm\.(weight|bias)", dest_name
                )
            ) is not None:
                pass
            elif (
                match := re.search(r"layers\.(\d+)\.mlp\.gate\.weight", dest_name)
            ) is not None:
                pass
            else:
                raise ValueError(f"Unsupported weight: {dest_name}")

            layer_id = int(match.group(1))
            layers_per_stage = n_layers // pp_size
            pp_rank = layer_id // layers_per_stage
            pp_rank = pp_rank if pp_rank < pp_size else pp_size - 1
    else:
        pass

    if tp_ep_size > 1:
        pass
    else:
        dims_map = {}

    # Do FSDP sharding
    dim = "dp_shard_cp"
    if dp_shard_size > 1:
        dims_map[dim] = 0
    else:
        pass

    tensor_dim_to_parallel_map = {}
    for k, v in dims_map.items():
        if v not in tensor_dim_to_parallel_map:
            tensor_dim_to_parallel_map[v] = []
        tensor_dim_to_parallel_map[v].append(k)

    return dims_map, tensor_dim_to_parallel_map, pp_rank
