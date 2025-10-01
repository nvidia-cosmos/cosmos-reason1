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

"""SFT adapter for llava-format datasets."""

import argparse
import json
import os
import re
from pathlib import Path

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import pydantic
import toml
import torch.utils.data
from cosmos_rl.utils.logging import logger

from cosmos_reason1_utils.text import create_conversation
from cosmos_reason1_utils.vision import VisionConfig


class CustomDatasetConfig(pydantic.BaseModel):
    annotation_path: str = pydantic.Field()
    """Dataset annotation path."""
    media_path: str = pydantic.Field(default="")
    """Dataset media path."""
    system_prompt: str = pydantic.Field(default="")
    """System prompt for post-training."""


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
        config: cosmos_rl.policy.config.Config,
        custom_config: CustomConfig,
    ):
        self.annotation = json.load(open(custom_config.dataset.annotation_path))
        self.media_path = custom_config.dataset.media_path
        self.system_prompt = custom_config.dataset.system_prompt
        self.config = config
        self.custom_config = custom_config
        self.vision_kwargs = custom_config.vision.model_dump(exclude_none=True)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx: int) -> list[dict]:
        sample = self.annotation[idx]

        user_prompt = sample["conversations"][0]["value"]
        response = sample["conversations"][1]["value"]
        images = sample.get("image", None) or sample.get("images", None)
        if images and isinstance(images, str):
            images = [images]
        videos = sample.get("video", None)
        if videos and isinstance(videos, str):
            videos = [videos]

        # If self.media_path is not empty, join it with each image/video path
        if self.media_path != "":
            if images:
                images = [os.path.join(self.media_path, img) for img in images]
            if videos:
                videos = [os.path.join(self.media_path, vid) for vid in videos]

        # Remove image and video tags from user prompt
        user_prompt = re.sub(r"(\n)?</?(image|video)>(\n)?", "", user_prompt)

        conversations = create_conversation(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response=response,
            images=images,
            videos=videos,
            vision_kwargs=self.vision_kwargs,
        )
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
        config=config,
        custom_config=custom_config,
    )
    # Check dataset
    print(dataset[0])

    # Launch worker
    cosmos_rl.launcher.worker_entry.main(
        dataset=dataset,
    )
