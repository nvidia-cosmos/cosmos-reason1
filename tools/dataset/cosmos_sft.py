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
import copy
import torch.utils.data
import datasets
import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import argparse
import toml

# Downsampling parameters
# Used by https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
FPS = 1
MAX_PIXELS = 81920


class CosmosSFTDataset(torch.utils.data.Dataset):
    """Dataset adapter for using huggingface datasets with SFT trainer."""

    def __init__(
        self, dataset: datasets.Dataset, config: cosmos_rl.policy.config.Config
    ):
        self.dataset = dataset
        self.config = config

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
                    if "image" in msg:
                        col_name = msg["image"]
                        msg["image"] = sample[col_name]
                        msg["max_pixels"] = MAX_PIXELS
                    if "video" in msg:
                        col_name = msg["video"]
                        video = sample[col_name]
                        if isinstance(self.dataset.features.get(col_name), datasets.Video):
                            video = video["path"]
                            assert video is not None
                        msg["video"] = video
                        msg["fps"] = FPS
                        msg["max_pixels"] = MAX_PIXELS
        return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = cosmos_rl.policy.config.Config.from_dict(toml.load(f))

    def get_dataset(config: cosmos_rl.policy.config.Config) -> torch.utils.data.Dataset:
        dataset_config = config.train.train_policy.dataset
        if dataset_config.name.startswith("file://"):
            dataset = datasets.load_from_disk(dataset_config.name.lstrip("file://"))
        else:
            dataset = datasets.load_dataset(
                dataset_config.name,
                dataset_config.subset or None,
            )
            if dataset_config.split:
                dataset = dataset[dataset_config.split]
        assert isinstance(dataset, datasets.Dataset)
        for col_name, col in dataset.features.items():
            if isinstance(col, datasets.Video):
                dataset = dataset.cast_column(col_name, datasets.Video(decode=False))
        return CosmosSFTDataset(dataset, config=config)

    # Test
    dataset = get_dataset(config)
    dataset[0]

    cosmos_rl.launcher.worker_entry.main(
        dataset=get_dataset,
    )
