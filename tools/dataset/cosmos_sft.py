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

"""Dataset adapter for using huggingface datasets with SFT trainer."""

import json
import copy
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
import cosmos_rl.utils.util as util
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.config import Config as CosmosConfig
from transformers import AutoTokenizer
import argparse
import toml

# Used by https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
FPS = 1
MAX_PIXELS = 81920

class CosmosSFTDataset(Dataset):
    """Dataset adapter for using huggingface datasets with SFT trainer."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        self.config = config
        self.tokenizer = tokenizer

        if config.train.train_policy.dataset.split:
            if isinstance(config.train.train_policy.dataset.split, list):
                dataset_list = []
                for split_name in config.train.train_policy.dataset.split:
                    dataset_list.append(self.dataset[split_name])
                self.dataset = ConcatDataset(dataset_list)
            else:
                assert isinstance(config.train.train_policy.dataset.split, str)
                self.dataset = self.dataset[config.train.train_policy.dataset.split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        """
        Return a tuple of (prompt, reference answer)
        """
        payload = self.dataset[idx]
        conversations = copy.deepcopy(payload[self.config.train.train_policy.conversation_column_name])
        for conv in conversations:
            if conv["role"] == "user":
                conv["content"] = json.loads(conv["content"])
                assert isinstance(conv["content"], list), "User messages must be a list"
                for msg in conv["content"]:
                    if "image" in msg:
                        msg["image"] = payload[msg["image"]]
                        msg["max_pixels"] = MAX_PIXELS
                    if "video" in msg:
                        msg["video"] = payload[msg["video"]]
                        msg["fps"] = FPS
                        msg["max_pixels"] = MAX_PIXELS

        return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = toml.load(f)
    config = Config.from_dict(config)

    def get_dataset(config: CosmosConfig) -> Dataset:
        dataset = load_dataset(
            config.train.train_policy.dataset.name,
            config.train.train_policy.dataset.subset,
        )
        return CosmosSFTDataset(dataset)

    # It is best practice to pass the dataset as a factory function
    # so that the dataset can be loaded on demand. (Not all workers need it)
    launch_worker(
        dataset=get_dataset,
    )
