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

from cosmos_reason1.policy.model.gpt import GPT
from cosmos_reason1.policy.model.qwen2_5_vl import Qwen2_5_VLConditionalModel
from cosmos_reason1.policy.model.qwen3_moe import Qwen3MoE
from cosmos_reason1.policy.config import Config as CosmosConfig
import cosmos_reason1.utils.util as util
from cosmos_reason1.utils.logging import logger
from transformers import AutoConfig
import torch


supported_cls_list = [GPT, Qwen2_5_VLConditionalModel, Qwen3MoE]


def get_data_packer(config: CosmosConfig):
    hf_config = util.retry(AutoConfig.from_pretrained)(
        config.policy.model_name_or_path, trust_remote_code=True
    )
    for model_cls in supported_cls_list:
        if hf_config.model_type in model_cls.supported_model_types():
            return model_cls.data_packer()
    raise ValueError(f"Model {config.policy.model_name_or_path} not supported.")


def build_model(config: CosmosConfig):
    model_name_or_path = config.policy.model_name_or_path
    model = None
    hf_config = util.retry(AutoConfig.from_pretrained)(
        model_name_or_path, trust_remote_code=True
    )

    with torch.device("meta"):
        with util.cosmos_default_dtype(config.train.param_torch_dtype):
            for model_cls in supported_cls_list:
                if hf_config.model_type in model_cls.supported_model_types():
                    try:
                        model = model_cls.from_pretrained(
                            hf_config,
                            model_name_or_path,
                            max_position_embeddings=config.policy.model_max_length,
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to load model {model_name_or_path} with error: {e}"
                        )
                        raise e
                    break
    if model is None:
        raise ValueError(f"Model {model_name_or_path} not supported.")
    return model
