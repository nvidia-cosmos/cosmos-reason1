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

import json
from torch.utils.data import Dataset
from datasets import concatenate_datasets
from cosmos_reason1.policy.config import Config as CosmosConfig
from cosmos_reason1.utils.util import load_data_from_disk_or_hf
from cosmos_reason1.utils.logging import logger
from typing import Optional, Any


# TODO(dinghaoy): Refactor and abstract a more general dataset loading and processing logic
class RLDataset(Dataset):
    def __init__(
        self,
        dataset: Any,
        prompt_column: str,
        response_column: Optional[str] = None,
        is_cosmos: bool = False,
        is_dapo_math: bool = False,
    ):
        self.dataset = dataset
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.is_cosmos = is_cosmos
        self.is_dapo_math = is_dapo_math

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[int, str]:
        if self.is_cosmos:
            return idx, json.dumps(self.dataset[idx])
        if self.is_dapo_math:
            return idx, self.dataset[idx][self.prompt_column][0]["content"]
        return idx, self.dataset[idx][self.prompt_column]

    def query_reference_answer(self, idx: int) -> str:
        if self.is_cosmos:
            return self.dataset[idx]["qa_pairs"][self.response_column]
        if self.is_dapo_math:
            return self.dataset[idx][self.response_column]["ground_truth"]
        return self.dataset[idx][self.response_column] if self.response_column else None


class CosmosDataset:
    def __init__(self, config: CosmosConfig):
        self.config = config
        self.grpo_config = config.train.train_policy
        self.is_dapo_math = False
        # Hack for cosmos RL dataset
        if self.grpo_config.prompt_column_name == "qa_pairs":
            self.is_cosmos = True
            self.prompt_column = "question"
            self.response_column = "answer"
            self.choices_column = "index2ans"
        else:
            # TODO(dinghaoy): Temporarily hack for specific datasets, will be unified by abstract dataset
            self.is_cosmos = False
            if "dapo-math" in self.grpo_config.dataset_name.lower():
                self.is_dapo_math = True
            self.prompt_column = self.grpo_config.prompt_column_name
            self.response_column = self.grpo_config.response_column_name
            self.choices_column = self.grpo_config.choices_column_name
        dataset = load_data_from_disk_or_hf(
            self.grpo_config.dataset_name,
            self.grpo_config.dataset_subset,
            self.grpo_config.dataset_revision or None,
        )

        dataset_list = []
        for split_name in self.grpo_config.dataset_train_split:
            logger.info(
                f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
            )
            dataset_list.append(dataset[split_name])
        train_dataset = concatenate_datasets(dataset_list)
        logger.info(f"Final dataset size = {len(train_dataset)}")

        self.train_set = RLDataset(
            train_dataset,
            self.prompt_column,
            self.response_column,
            self.is_cosmos,
            self.is_dapo_math,
        )
