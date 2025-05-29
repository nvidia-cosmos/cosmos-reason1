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

from cosmos_reason1.rollout.weight_mapper.registry import register_class, WeightMapper
from transformers import AutoConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
)
from cosmos_reason1.policy.model.qwen2_5_vl import Qwen2_5_VLConditionalModel
import torch
import copy
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from cosmos_reason1.utils.parallelism import ParallelismConfig
from cosmos_reason1.utils.parallelism_registry import (
    get_policy_parallelism_strategy,
    get_rollout_parallelism_strategy,
)
import cosmos_reason1.utils.util as util


@register_class("qwen2_5_vl")
class QwenVL25WeightMapper(WeightMapper):
    def __init__(self, hf_config_path: str):
        super().__init__(hf_config_path)
        self.qwen_config = util.retry(AutoConfig.from_pretrained)(hf_config_path)
        self.prefix_str = None
        assert isinstance(self.qwen_config, Qwen2_5_VLConfig)

        self.kv_head_ratio = (
            self.qwen_config.num_attention_heads // self.qwen_config.num_key_value_heads
        )
        self.head_dim = (
            self.qwen_config.hidden_size // self.qwen_config.num_attention_heads
        )

    def name_p2r(self, policy_weight_name: str) -> str:
        raise NotImplementedError("QwenVL25WeightMapper does not support name_p2r")

    def name_to_hf(self, rollout_weight_name: str) -> str:
        # convert from rollout weight name to policy weight name
        return Qwen2_5_VLConditionalModel.map_local_key_to_hf_key(rollout_weight_name)

    def split_qkv_weight(self, name, weight: torch.Tensor):
        # visual
        if "visual" in name:
            # split qkv weight for visual
            # weight has shape [3 * head_dim, hidden_dim]
            # kv head ratio is 1, so we can split it into q, k, v
            assert (
                weight.shape[0] % 3 == 0
            ), "Weight shape is not compatible for splitting."
            unit_dim = weight.shape[0] // 3  # for both weight and bias
            q_weight = weight[:unit_dim]
            k_weight = weight[unit_dim : unit_dim * 2]
            v_weight = weight[unit_dim * 2 :]
            return q_weight, k_weight, v_weight
        # language
        # split qkv weight
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
        # gate_proj and up_proj in vllm is already split.
        # weight has shape [2 * x, hidden_dim]
        dim_0 = weight.shape[0]
        gate_proj_weight = weight[: dim_0 // 2]
        up_proj_weight = weight[dim_0 // 2 :]
        return gate_proj_weight, up_proj_weight

    def custom_generate_compatible_map(self, model: Qwen2_5_VLForConditionalGeneration):
        assert isinstance(model, Qwen2_5_VLForConditionalGeneration)
        compatible_list = []
        compatible_weight_map = {}
        for param_name, param in model.named_parameters():
            compatible_key = self.name_to_hf(param_name)
            if "qkv_proj" in compatible_key:
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
            elif "qkv" in compatible_key and "visual" in compatible_key:
                q_weight, k_weight, v_weight = self.split_qkv_weight(
                    compatible_key, param
                )
                q_visual_proj_weight_key = compatible_key.replace("qkv", "q")
                k_visual_proj_weight_key = compatible_key.replace("qkv", "k")
                v_visual_proj_weight_key = compatible_key.replace("qkv", "v")
                compatible_weight_map[q_visual_proj_weight_key] = q_weight
                compatible_list.append((q_visual_proj_weight_key, q_weight.shape))
                compatible_weight_map[k_visual_proj_weight_key] = k_weight
                compatible_list.append((k_visual_proj_weight_key, k_weight.shape))
                compatible_weight_map[v_visual_proj_weight_key] = v_weight
                compatible_list.append((v_visual_proj_weight_key, v_weight.shape))
            else:
                compatible_weight_map[compatible_key] = param
                compatible_list.append((compatible_key, param.shape))
        return compatible_weight_map, compatible_list

    def name_to_model_index(self, dest_name: str) -> int:
        if "lm_head.weight" == dest_name:
            return 0
        elif "lm_head.bias" == dest_name:
            return 0
        elif dest_name.startswith("visual."):
            return 1
        elif dest_name.startswith("model."):
            return 0
        else:
            raise ValueError(f"Unsupported weight: {dest_name}")

    def get_rollout_parallelism(self, parallelism_config: ParallelismConfig):
        return [parallelism_config, parallelism_config]

    def get_policy_parallelism(self, parallelism_config: ParallelismConfig):
        parallelism_config_new = copy.copy(parallelism_config)
        assert parallelism_config.tp_size != -1
        if parallelism_config_new.dp_shard_size != -1:
            parallelism_config_new.dp_shard_size = (
                parallelism_config.dp_shard_size * parallelism_config.tp_size
            )
        parallelism_config_new.tp_size = 1
        return [parallelism_config, parallelism_config_new]

    def get_policy_parallelism_strategy(self):
        return [
            get_policy_parallelism_strategy("gpt"),
            get_policy_parallelism_strategy("qwen2_5_vl"),
        ]

    def get_rollout_parallelism_strategy(self):
        return [
            get_rollout_parallelism_strategy("gpt"),
            get_rollout_parallelism_strategy("qwen2_5_vl"),
        ]
