#!/usr/bin/env -S uv run --script
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

#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cosmos-reason1-utils",
#   "datasets",
#   "pyyaml",
#   "rich",
#   "tqdm",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# [tool.uv.sources]
# cosmos-reason1-utils = {path = "../../../cosmos_reason1_utils", editable = true}
# ///

"""Download Nexar collision prediction dataset.

https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction
"""

import argparse
import json
from pathlib import Path

import datasets
import yaml
from rich import print
from tqdm import tqdm

from cosmos_reason1_utils.text import PromptConfig, create_conversation

ROOT = Path(__file__).parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=str, help="Output huggingface dataset path.")
    parser.add_argument("--split", type=str, default="train", help="Split to download.")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt
    system_prompt = PromptConfig.model_validate(
        yaml.safe_load(open(f"{ROOT}/prompts/question.yaml", "rb"))
    ).system_prompt
    user_prompt = "What is the weather in this video? Choose from ['Rain', 'Cloudy', 'Snow', 'Clear']."

    # Load raw dataset
    dataset = datasets.load_dataset(
        "nexar-ai/nexar_collision_prediction", split=args.split
    )
    print(dataset)
    dataset = dataset.cast_column("video", datasets.Video(decode=False))
    dataset_size = len(dataset)

    # Save training dataset
    def process_sample(sample: dict) -> dict:
        # Store media separately
        video_path = sample["video"]["path"]
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        conversation = create_conversation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            videos=[video_path],
            response=str(sample["weather"]),
        )
        return {
            # Store conversation as string
            "conversations": json.dumps(conversation),
        }

    dataset = list(
        tqdm(
            map(process_sample, dataset), total=dataset_size, desc="Processing dataset"
        )
    )
    dataset = datasets.Dataset.from_generator(lambda: dataset)
    print(dataset)
    dataset.save_to_disk(str(output_dir))


if __name__ == "__main__":
    main()
