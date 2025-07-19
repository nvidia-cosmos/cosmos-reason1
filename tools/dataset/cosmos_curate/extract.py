"""Extract cosmos-curate webdataset into huggingface dataset."""

import argparse
import os
from pathlib import Path
from typing import  Iterable
import webdataset
import datasets
from rich import print

def replace_column_name(url: str, old_name: str, new_name: str) -> str:
    if url.count(f"/{old_name}/") != 1:
        raise ValueError(f"Name {old_name} not found in {url}")
    return url.replace(f"/{old_name}/", f"/{new_name}/")


def get_shard_urls(dataset_path: str, *, column_name: str = "metas") -> list[str]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
    tar_files = []
    for root, _dirs, files in os.walk(dataset_path):
        if f"/{column_name}/" not in root:
            continue
        for file in files:
            if not file.endswith(".tar"):
                continue
            tar_files.append(os.path.join(root, file))
    tar_files.sort()
    if not tar_files:
        raise FileNotFoundError(f"No tar files found in {dataset_path}")
    return tar_files


def join_webdatasets(datasets: list[Iterable], column_names: list[str]) -> Iterable:
    for cols in zip(*datasets, strict=True):
        # Check that all keys are the same
        key_0 = cols[0]["__key__"]
        for col in cols[1:]:
            key = col["__key__"]
            if key != key_0:
                raise ValueError(f"Key mismatch: {key} != {key_0}")

        sample = dict(zip(column_names, cols, strict=True))
        sample["__key__"] = key_0
        yield sample


def load_webdataset(
    dataset_path: str, column_names: list[str], dataset_kwargs: dict | None = None
) -> Iterable:
    if not column_names:
        raise ValueError("Must specify at least one column name")
    if dataset_kwargs is None:
        dataset_kwargs = {}
    dataset_kwargs = (
        dict(
            shardshuffle=False,
        )
        | dataset_kwargs
    )

    col_name_0 = column_names[0]
    urls_0 = get_shard_urls(dataset_path, column_name=col_name_0)
    datasets = []
    for col_name in column_names:
        urls = [replace_column_name(url, col_name_0, col_name) for url in urls_0]
        datasets.append(webdataset.WebDataset(urls, **dataset_kwargs))
    return join_webdatasets(datasets, column_names)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Input directory containing cosmos-curate webdataset.")
    parser.add_argument("output", type=str, help="Output directory to save huggingface dataset.")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load
    dataset = load_webdataset(args.input, ["metas", "video"])

    # Process
    def process_sample(sample: dict) -> dict:
        key = sample["__key__"]

        # Save media to disk and only store path in dataset
        video_path = output_dir / "media" / key / "video.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(sample["video"]["mp4"])

        return {
            "__key__": key,
            "metas": sample["metas"]["json"],
            "video": str(video_path),
        }

    dataset = list(map(process_sample, dataset))
    dataset = datasets.Dataset.from_generator(lambda: dataset)
    print(dataset)
    dataset.save_to_disk(str(output_dir))

if __name__ == "__main__":
    main()