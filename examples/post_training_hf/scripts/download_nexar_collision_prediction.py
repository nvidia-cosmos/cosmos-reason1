#!/usr/bin/env -S uv run --script
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
from tqdm import tqdm
from rich import print
import yaml

from cosmos_reason1_utils.text import create_conversation, PromptConfig

ROOT = Path(__file__).parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=str, help="Output huggingface dataset path.")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt
    system_prompt = PromptConfig.model_validate(yaml.safe_load(open(f"{ROOT}/prompts/question.yaml", "rb"))).system_prompt
    user_prompt = "What is the weather in this video? Choose from ['Rain', 'Cloudy', 'Snow', 'Clear']."

    # Load raw dataset
    dataset = datasets.load_dataset(
        "nexar-ai/nexar_collision_prediction", split="train"
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
