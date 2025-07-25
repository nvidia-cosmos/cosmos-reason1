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

"""Dataset adapter for huggingface datasets for SFT training cosmos models."""

import enum
import json
import copy
from typing import assert_never
import torch.utils.data
import datasets
import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import argparse
import toml
import webdataset
import pydantic


class DatasetType(str, enum.Enum):
    """Dataset type for SFT training."""

    HUGGINGFACE_HUB = "huggingface_hub"
    """Dataset generated using `datasets.Dataset.push_to_hub`."""
    HUGGINGFACE_DISK = "huggingface_disk"
    """Dataset generated using `datasets.Dataset.save_to_disk`."""
    WEBDATASET = "webdataset"
    """Dataset generated using `webdataset`."""


class CosmosSFTConfig(pydantic.BaseModel):
    """Config for Cosmos SFT training."""

    dataset_type: DatasetType = pydantic.Field()
    """Dataset type."""

    fps: int = pydantic.Field(default=1)
    """Downsample video frame rate."""
    max_pixels: int = pydantic.Field(default=81920)
    """Downsample image/video max pixels per frame."""


class CosmosSFTDataset(torch.utils.data.Dataset):
    """Dataset adapter for using huggingface datasets with SFT trainer."""

    def __init__(
        self, dataset: datasets.Dataset, config: cosmos_rl.policy.config.Config, custom_config: CosmosSFTConfig
    ):
        self.dataset = dataset
        self.config = config
        self.custom_config = custom_config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list[dict]:
        """
        Return a tuple of (prompt, reference answer)
        """
        sample = self.dataset[idx]
        conversations = copy.deepcopy(
            sample[self.config.train.train_policy.conversation_column_name]
        )
        for conv in conversations:
            if conv["role"] == "user":
                conv["content"] = json.loads(conv["content"])
                assert isinstance(conv["content"], list), "User messages must be a list"
                for msg in conv["content"]:
                    # Set parameters for https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
                    if "image" in msg:
                        col_name = msg["image"]
                        msg["image"] = sample[col_name]
                        msg["max_pixels"] = self.custom_config.max_pixels
                    if "video" in msg:
                        col_name = msg["video"]
                        video = sample[col_name]
                        if isinstance(
                            self.dataset.features.get(col_name), datasets.Video
                        ):
                            video = video["path"]
                            assert video is not None
                        msg["video"] = video
                        msg["fps"] = self.custom_config.fps
                        msg["max_pixels"] = self.custom_config.max_pixels
        return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = cosmos_rl.policy.config.Config.from_dict(toml.load(f))

    def get_dataset(config: cosmos_rl.policy.config.Config) -> torch.utils.data.Dataset:
        custom_config = CosmosSFTConfig.model_validate(config.custom)
        dataset_config = config.train.train_policy.dataset

        if custom_config.dataset_type == DatasetType.HUGGINGFACE_HUB:
            dataset = datasets.load_dataset(
                dataset_config.name,
                dataset_config.subset or None,
                split=dataset_config.split or None,
            )
        elif custom_config.dataset_type == DatasetType.HUGGINGFACE_DISK:
            dataset = datasets.load_from_disk(dataset_config.name)
        elif custom_config.dataset_type == DatasetType.WEBDATASET:
            dataset = webdataset.WebDataset(
                dataset_config.name.lstrip("webdataset://"), shardshuffle=False
            )
        else:
            assert_never(custom_config.dataset_type)
        for col_name, col in dataset.features.items():
            if isinstance(col, datasets.Video):
                dataset = dataset.cast_column(col_name, datasets.Video(decode=False))
        return CosmosSFTDataset(dataset, config=config, custom_config=custom_config)

    # Test
    dataset = get_dataset(config)
    dataset[0]

    cosmos_rl.launcher.worker_entry.main(
        dataset=get_dataset,
    )
