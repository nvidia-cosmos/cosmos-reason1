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

from torch.utils.data import Dataset
from datasets import concatenate_datasets
from cosmos_reason1.policy.config import Config as CosmosConfig
from cosmos_reason1.utils.util import load_data_from_disk_or_hf
from cosmos_reason1.utils.logging import logger
from typing import Optional, Any, List
from transformers import AutoTokenizer


class RLPayload:
    payload: Any

    def __init__(self, payload: Any):
        self.payload = payload

    @staticmethod
    def collate_fn(
        batch: List[tuple[int, "RLPayload", str]],
    ) -> tuple[List[int], List["RLPayload"], List[str]]:
        return (
            [item[0] for item in batch],
            [item[1] for item in batch],
        )


class RLDataset(Dataset):
    def __init__(self, dataset: Any, tokenizer: AutoTokenizer, config: CosmosConfig):
        self.dataset = dataset
        if hasattr(self.dataset, "setup"):
            self.dataset.setup(tokenizer=tokenizer, config=config)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[int, RLPayload, str]:
        payload = self.dataset[idx]
        return idx, RLPayload(payload)

    def get_reference_answer(self, idx: int) -> Any:
        assert hasattr(
            self.dataset, "get_reference_answer"
        ), "Dataset should have a `get_reference_answer` method"
        return self.dataset.get_reference_answer(idx)


class RLInternalDataset(Dataset):
    def __init__(
        self,
        dataset: Any,
        prompt_column: str,
        response_column: Optional[str] = None,
    ):
        self.dataset = dataset
        self.prompt_column = prompt_column
        self.response_column = response_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[int, RLPayload, str]:
        payload = self.dataset[idx][self.prompt_column]
        return idx, RLPayload(payload)

    def get_reference_answer(self, idx: int) -> Any:
        ref = self.dataset[idx][self.response_column]
        return ref


class CosmosDataset:
    def __init__(
        self,
        config: CosmosConfig,
        train_set: Optional[Dataset] = None,
        tokenizer: AutoTokenizer = None,
    ):
        self.config = config
        if train_set is not None:
            self.train_set = RLDataset(train_set, tokenizer, config)
        else:
            """
            Deprecated: for most cases, users should provide a train_set for better generalization
            """
            self.grpo_config = config.train.train_policy
            self.prompt_column = self.grpo_config.prompt_column_name
            self.response_column = self.grpo_config.response_column_name
            dataset = load_data_from_disk_or_hf(
                self.grpo_config.dataset.name,
                self.grpo_config.dataset.subset,
                self.grpo_config.dataset.revision or None,
            )

            dataset_list = []
            for split_name in self.grpo_config.dataset.split:
                logger.info(
                    f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
                )
                dataset_list.append(dataset[split_name])
            train_dataset = concatenate_datasets(dataset_list)

            logger.info(f"Final dataset size = {len(train_dataset)}")
            self.train_set = RLInternalDataset(
                train_dataset,
                self.prompt_column,
                self.response_column,
            )


class CosmosValidationDataset:
    def __init__(
        self,
        config: CosmosConfig,
        val_set: Optional[Dataset] = None,
        tokenizer: AutoTokenizer = None,
    ):
        self.config = config
        if val_set is not None:
            self.val_set = RLDataset(val_set, tokenizer, config)
        else:
            """
            Deprecated: for most cases, users should provide a train_set for better generalization
            """
            self.grpo_config = config.train.train_policy
            self.prompt_column = self.grpo_config.prompt_column_name
            self.response_column = self.grpo_config.response_column_name
            dataset = load_data_from_disk_or_hf(
                self.config.validation.dataset.name,
                self.config.validation.dataset.subset,
                self.config.validation.dataset.revision or None,
            )
            dataset_list = []
            for split_name in self.config.validation.dataset.split:
                logger.info(
                    f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
                )
                dataset_list.append(dataset[split_name])
            val_dataset = concatenate_datasets(dataset_list)

            self.val_set = RLInternalDataset(
                val_dataset,
                self.prompt_column,
                self.response_column,
            )
