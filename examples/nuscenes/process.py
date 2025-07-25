"""Convert webdataset to huggingface disk dataset."""

import argparse
import collections
import json
import os
from pathlib import Path
import shutil
from typing import Any
import torch
import webdataset
import datasets

def load_webdataset(
    dataset_path: str, dataset_kwargs: dict = {}
) -> torch.utils.data.Dataset:
    tar_files = []
    for root, _dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".tar"):
                tar_files.append(os.path.join(root, file))
    tar_files.sort()
    return webdataset.WebDataset(
        tar_files,
        **(
            dict(
                shardshuffle=False,
            )
            | dataset_kwargs
        ),
    )

def _update_dict(d: dict, key: str, val: Any):
    """Update a dictionary with new key-value pairs.
    
    If the key already exists, raise an error if the value is different.
    """
    if key in d:
        if d[key] != val:
            raise ValueError(f"Duplicate key {key} with different values {d[key]} != {val}")
    else:
        d[key] = val

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=str, required=True, help="Input directory.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory.")
    parser.add_argument("--meta", type=str, nargs="+", help="Meta column names.")
    parser.add_argument("--media", type=str, nargs="+", help="Media column names.")
    args = parser.parse_args()

    meta_names: frozenset[str] = frozenset(args.meta)
    media_names: frozenset[str] = frozenset(args.media)
    if meta_names & media_names:
        raise ValueError("Meta and media column names must be disjoint")

    output_dir = Path(args.output)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_from_disk(args.input)

    def add_conversations(sample: dict) -> dict:
        key = sample["__key__"]
        caption = sample["windows"][0]["qwen_caption"]
        parts = caption.split("\n-\n")
        if len(parts) != 2:
            raise ValueError(f"Invalid caption for sample {key}: {caption}")
        question, answer = parts
        conversations = [
            {
                    "role": "system",
                    "content": "You are a helpful assistant. Please answer the questions.",
                },
                {
                    "role": "user",
                    "content": json.dumps([
                        {
                            "type": "video",
                            "video": "mp4",
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ]),
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
        ]
        return {
            "conversations": conversations,
        }
    dataset = dataset.map(add_conversations)
    print(dataset)
    dataset.save_to_disk(str(output_dir))

if __name__ == "__main__":
    main()