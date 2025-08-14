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

"""SFT adapter for huggingface datasets."""

import argparse
import json

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import datasets
import pydantic
import toml
import torch.utils.data
from cosmos_reason1_utils.text import set_vision_kwargs
from cosmos_reason1_utils.vision import VisionConfig


class CustomDatasetConfig(pydantic.BaseModel):
    path: str = pydantic.Field()
    """Dataset path."""


class CustomConfig(pydantic.BaseModel):
    dataset: CustomDatasetConfig = pydantic.Field()
    """Dataset config."""

    vision: VisionConfig = pydantic.Field(
        default=VisionConfig(
            fps=1,
            max_pixels=81920,
        )
    )
    """Vision processor config."""


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        config: cosmos_rl.policy.config.Config,
        custom_config: CustomConfig,
    ):
        self.dataset = dataset
        self.config = config
        self.custom_config = custom_config
        self.vision_kwargs = custom_config.vision.model_dump(exclude_none=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list[dict]:
        sample = self.dataset[idx]
        conversations = json.loads(
            sample[self.config.train.train_policy.conversation_column_name]
        )
        set_vision_kwargs(conversations, self.vision_kwargs)
        return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    args = parser.parse_known_args()[0]

    with open(args.config, "r") as f:
        config = cosmos_rl.policy.config.Config.from_dict(toml.load(f))

    def get_dataset(config: cosmos_rl.policy.config.Config) -> torch.utils.data.Dataset:
        custom_config = CustomConfig.model_validate(config.custom)
        dataset = datasets.load_from_disk(
            custom_config.dataset.path,
        )
        return CustomDataset(dataset, config=config, custom_config=custom_config)

    # Check dataset
    dataset = get_dataset(config)
    dataset[0]

    # Launch worker
    cosmos_rl.launcher.worker_entry.main(
        dataset=get_dataset,
    )
