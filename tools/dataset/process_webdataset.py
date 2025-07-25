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
    parser.add_argument("input", type=str, help="Input directory containing webdataset.")
    parser.add_argument("output", type=str, help="Output directory to save huggingface dataset.")
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
    
    # Load the webdataset
    dataset = load_webdataset(args.input)

    # Join columns by key
    samples: dict[str, dict] = {}
    for other in dataset:
        key: str = other["__key__"]
        if key not in samples:
            samples[key] = {"__key__": key}
        sample = samples[key]
        for name, val in other.items():
            if name in media_names:
                # Save media to disk and only store path in dataset
                path = output_dir / "media" / key / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(val)
                val = str(path)
                _update_dict(sample, name, val)
            elif name in meta_names:
                val = json.loads(val)
                _update_dict(sample, name, val)

    # Create huggingface dataset
    names = meta_names | media_names | {"__key__"}
    data = {name: [] for name in names}
    for sample in samples.values():
        key = sample["__key__"]
        for name in names:
            if name not in sample:
                raise ValueError(f"Missing column {name} for sample {key}")
            data[name].append(sample[name])
    dataset = datasets.Dataset.from_dict(data)
    print(dataset)

    dataset.save_to_disk(str(output_dir))

if __name__ == "__main__":
    main()