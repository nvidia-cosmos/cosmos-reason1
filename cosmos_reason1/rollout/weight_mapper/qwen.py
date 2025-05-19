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

from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from cosmos_reason1.policy.model.gpt import GPT
import torch
from transformers import AutoConfig
from typing import List, Tuple, Dict
from cosmos_reason1.rollout.weight_mapper.registry import (
    WeightMapper,
    register_class,
)
from cosmos_reason1.utils.parallelism import ParallelismConfig
from cosmos_reason1.utils.parallelism_registry import (
    get_policy_parallelism_strategy,
    get_rollout_parallelism_strategy,
)


@register_class("qwen2")
@register_class("qwen3")
class QwenWeightMapper(WeightMapper):
    def __init__(self, hf_config_path: str):
        self.prefix_str = "model."
        self.tp_size = 2
        self.qwen_config = AutoConfig.from_pretrained(hf_config_path)
        assert (
            "qwen" in type(self.qwen_config).__name__.lower()
        ), f"qwen_config is not a QwenConfig: {type(self.qwen_config).__name__}"
        self.kv_head_ratio = (
            self.qwen_config.num_attention_heads // self.qwen_config.num_key_value_heads
        )
        self.head_dim = (
            self.qwen_config.hidden_size // self.qwen_config.num_attention_heads
        )

    def name_p2r(self, policy_weight_name: str) -> str:
        rollout_weight_name = self.prefix_str + policy_weight_name
        return rollout_weight_name

    def name_to_hf(self, rollout_weight_name: str) -> str:
        return GPT.map_local_key_to_hf_key(rollout_weight_name)

    def split_qkv_weight(self, name, weight: torch.Tensor):
        # weight has shape [q_num_heads * head_dim + k_num_heads * head_dim + v_num_heads * head_dim, hidden_dim]
        shares = self.kv_head_ratio + 2
        dim_0 = weight.shape[0]  # for both weight and bias
        unit_dim = dim_0 // shares

        q_weight = weight[: unit_dim * self.kv_head_ratio]
        k_weight = weight[
            unit_dim * self.kv_head_ratio : unit_dim * (self.kv_head_ratio + 1)
        ]
        v_weight = weight[unit_dim * (self.kv_head_ratio + 1) :]
        return q_weight, k_weight, v_weight

    def split_gate_proj_weight(self, name, weight: torch.Tensor):
        # weight has shape [2 * x, hidden_dim]
        dim_0 = weight.shape[0]
        gate_proj_weight = weight[: dim_0 // 2]
        up_proj_weight = weight[dim_0 // 2 :]
        return gate_proj_weight, up_proj_weight

    def custom_generate_compatible_map(
        self, model: Qwen2ForCausalLM
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, torch.Size]]]:
        assert (
            "qwen" in type(model).__name__.lower()
        ), f"model is not a QwenForCausalLM: {type(model).__name__}"
        compatible_list = []
        compatible_weight_map = {}
        for param_name, param in model.named_parameters():
            compatible_key = self.name_to_hf(param_name)
            # logger.info(f"[Rollout] compatible_key: {compatible_key}")
            if "qkv_proj" in compatible_key:
                # must be inplace slicing.
                # split qkv weight
                q_weight, k_weight, v_weight = self.split_qkv_weight(
                    compatible_key, param
                )
                q_proj_weight_key = compatible_key.replace("qkv_proj", "q_proj")
                k_proj_weight_key = compatible_key.replace("qkv_proj", "k_proj")
                v_proj_weight_key = compatible_key.replace("qkv_proj", "v_proj")
                compatible_weight_map[q_proj_weight_key] = q_weight
                compatible_list.append((q_proj_weight_key, q_weight.shape))
                compatible_weight_map[k_proj_weight_key] = k_weight
                compatible_list.append((k_proj_weight_key, k_weight.shape))
                compatible_weight_map[v_proj_weight_key] = v_weight
                compatible_list.append((v_proj_weight_key, v_weight.shape))
            elif "gate_up_proj" in compatible_key:
                # split gate and up proj
                gate_proj_weight, up_proj_weight = self.split_gate_proj_weight(
                    compatible_key, param
                )
                gate_proj_weight_key = compatible_key.replace(
                    "gate_up_proj", "gate_proj"
                )
                compatible_weight_map[gate_proj_weight_key] = gate_proj_weight
                compatible_list.append((gate_proj_weight_key, gate_proj_weight.shape))

                up_proj_weight_key = compatible_key.replace("gate_up_proj", "up_proj")
                compatible_weight_map[up_proj_weight_key] = up_proj_weight
                compatible_list.append((up_proj_weight_key, up_proj_weight.shape))
            else:
                compatible_weight_map[compatible_key] = param
                compatible_list.append((compatible_key, param.shape))

        return compatible_weight_map, compatible_list

    def name_to_model_index(self, dest_name: str) -> int:
        return 0

    def get_rollout_parallelism(self, parallelism_config: ParallelismConfig):
        return [parallelism_config]

    def get_policy_parallelism(self, parallelism_config: ParallelismConfig):
        return [parallelism_config]

    def get_policy_parallelism_strategy(self):
        return [get_policy_parallelism_strategy("gpt")]

    def get_rollout_parallelism_strategy(self):
        return [get_rollout_parallelism_strategy("gpt")]
