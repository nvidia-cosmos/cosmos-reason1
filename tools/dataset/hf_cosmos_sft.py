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

import json
import torch.utils.data
import datasets
import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import argparse
import toml
import pydantic


class DatasetConfig(pydantic.BaseModel):
    """Config for dataset."""

    path: str = pydantic.Field()
    """Dataset path."""


class CosmosSFTConfig(pydantic.BaseModel):
    """Config for Cosmos SFT training."""

    dataset: DatasetConfig = pydantic.Field()
    """Dataset config."""

    fps: int = pydantic.Field(default=1)
    """Downsample video frame rate."""
    max_pixels: int = pydantic.Field(default=81920)
    """Downsample image/video max pixels per frame."""


class CosmosSFTDataset(torch.utils.data.Dataset):
    """Dataset adapter for using huggingface datasets with SFT trainer."""

    def __init__(
        self,
        dataset: datasets.Dataset,
        config: cosmos_rl.policy.config.Config,
        custom_config: CosmosSFTConfig,
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
        conversations = json.loads(
            sample[self.config.train.train_policy.conversation_column_name]
        )
        for conv in conversations:
            if conv["role"] == "user":
                content = conv["content"]
                if isinstance(content, str):
                    content = [content]
                for msg in content:
                    # Copy media and set parameters: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
                    if "image" in msg:
                        msg["image"] = sample[msg["image"]]
                        msg["max_pixels"] = self.custom_config.max_pixels
                    if "video" in msg:
                        msg["video"] = sample[msg["video"]]
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
        dataset = datasets.load_from_disk(
            custom_config.dataset.path,
        )
        return CosmosSFTDataset(dataset, config=config, custom_config=custom_config)

    # Test
    dataset = get_dataset(config)
    dataset[0]

    cosmos_rl.launcher.worker_entry.main(
        dataset=get_dataset,
    )
