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
import os
from pathlib import Path

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import datasets
import pydantic
import toml
import torch.utils.data
from cosmos_rl.utils.logging import logger

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
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file."
    )
    args = parser.parse_known_args()[0]

    # Load config
    with open(args.config) as f:
        config_kwargs = toml.load(f)
    config = cosmos_rl.policy.config.Config.from_dict(config_kwargs)
    custom_config = CustomConfig.model_validate(config_kwargs["custom"])

    # Log
    role = os.environ.get("COSMOS_ROLE")
    is_controller = role == "Controller"
    if is_controller:
        output_dir = Path(config.train.output_dir).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_kwargs = config.model_dump()
        config_kwargs["custom"] = custom_config.model_dump()
        config_path = output_dir / "config.toml"
        config_path.write_text(toml.dumps(config_kwargs))
        logger.info(f"Saved config to {config_path}")

    # Load dataset
    dataset = CustomDataset(
        datasets.load_from_disk(custom_config.dataset.path),
        config=config,
        custom_config=custom_config,
    )
    # Check dataset
    dataset[0]

    # Launch worker
    cosmos_rl.launcher.worker_entry.main(
        dataset=dataset,
    )
