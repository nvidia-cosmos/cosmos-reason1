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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets",
#   "ffmpeg-python",
#   "huggingface-hub",
#   "numpy",
#   "opencv-python",
#   "tensorflow_datasets",
#   "tqdm",
# ]
# [tool.uv]
# exclude-newer = "2025-08-05T00:00:00Z"
# ///

import argparse
import glob
import json
import logging as log
import os
import re
import subprocess
import tarfile
import urllib.request

import cv2
import ffmpeg
import numpy as np
import tensorflow_datasets as tfds
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, login
from tqdm import tqdm

"""Download and unpack datasets."""

log.basicConfig(level=log.INFO)

ALL_DATASETS = [
    "agibot",
    "bridgev2",
    "holoassist",
]


def tqdm_hook(t):
    """Wraps tqdm instance for use with urllib."""
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def download_agibot_data(data_dir: str, hf_token: str | None = None):
    """
    Download all data from the AgiBotWorld-Beta dataset on Hugging Face
    and extract any .tar files found.

    Args:
        target_dir: The target directory where files will be saved
        hf_token: Hugging Face API token for private repositories (if needed)

    Returns:
        List of paths to downloaded files
    """
    # Define the repository ID
    repo_id = "agibot-world/AgiBotWorld-Beta"
    repo_type = "dataset"
    dataset = "agibot"

    # Authenticate with Hugging Face
    if hf_token:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face")
    else:
        print("No Hugging Face token provided. Attempting to use cached credentials...")

    # Create target directory if it doesn't exist
    target_dir = os.path.join(data_dir, "raw", dataset)
    os.makedirs(target_dir, exist_ok=True)

    # Initialize the Hugging Face API
    api = HfApi(token=hf_token)

    try:
        print(f"Accessing repository: {repo_id}")

        # List repository contents
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

        # Count .tar files for information
        tar_files = [f for f in files if f.endswith(".tar")]
        other_files = [f for f in files if not f.endswith(".tar")]

        print(f"\nFound {len(files)} total files in the repository:")
        print(f"- {len(tar_files)} .tar files that will be extracted")
        print(f"- {len(other_files)} other files")

        if not files:
            print("\nNo files found in the repository.")
            return []

        downloaded_files = []

        # Download all files
        for i, file_path in enumerate(files):
            try:
                # Extract directory part from file path
                dir_parts = os.path.dirname(file_path).split("/")
                if dir_parts and dir_parts[0]:
                    subdir_path = os.path.join(target_dir, *dir_parts)
                    os.makedirs(subdir_path, exist_ok=True)
                else:
                    subdir_path = target_dir

                # Construct local file path
                file_name = os.path.basename(file_path)
                local_file_path = os.path.join(subdir_path, file_name)

                # Simple check if file already exists
                if os.path.exists(local_file_path):
                    print(
                        f"\n({i + 1}/{len(files)}) Skipping {file_path} as it already exists at {local_file_path}"
                    )
                    downloaded_files.append(local_file_path)

                    # Check if we should extract existing tar files that weren't extracted before
                    if file_path.endswith(".tar") and not os.path.exists(
                        os.path.join(subdir_path, ".extraction_completed")
                    ):
                        print(
                            f"Extracting existing file {file_name} to {subdir_path}..."
                        )
                        with tarfile.open(local_file_path, "r") as tar:
                            tar.extractall(path=subdir_path)
                        # Create marker file to avoid re-extraction in future runs
                        with open(
                            os.path.join(subdir_path, ".extraction_completed"), "w"
                        ) as f:
                            f.write("Extraction completed")
                        print(f"Extraction of existing {file_name} complete.")

                    continue

                print(f"\n({i + 1}/{len(files)}) Downloading {file_path}...")

                # Download file from Hugging Face
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    token=hf_token,
                    repo_type=repo_type,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                )

                downloaded_files.append(downloaded_file)

                # Extract if it's a .tar file
                if file_path.endswith(".tar"):
                    print(f"Extracting {file_name} to {subdir_path}...")
                    with tarfile.open(downloaded_file, "r") as tar:
                        tar.extractall(path=subdir_path)
                    # Create marker file to avoid re-extraction in future runs
                    with open(
                        os.path.join(subdir_path, ".extraction_completed"), "w"
                    ) as f:
                        f.write("Extraction completed")
                    print(f"Extraction of {file_name} complete.")

                print(f"Successfully processed {file_name}")
            except Exception as e:
                print(f"Error processing {file_path}: {e!s}")

        # Summary
        tar_downloads = sum(1 for f in downloaded_files if f.endswith(".tar"))
        other_downloads = len(downloaded_files) - tar_downloads

        print("\nDownload complete!")
        print(f"Downloaded {len(downloaded_files)} files to {target_dir}:")
        print(f"- {tar_downloads} .tar files (all extracted)")
        print(f"- {other_downloads} other files")

        return downloaded_files

    except Exception as e:
        print(f"Error accessing repository: {e!s}")
        print("Please verify the repository exists and your token has access.")
        return []


def download_holoassist_data(data_dir: str, dataset: str) -> tuple[str, str]:
    raw_dataset_path = os.path.join(data_dir, "raw", dataset)
    os.makedirs(raw_dataset_path, exist_ok=True)

    # Download metadata
    metadata_url = "https://hl2data.z5.web.core.windows.net/holoassist-data-release/data-annotation-trainval-v1_1.json"
    metadata_path = os.path.join(raw_dataset_path, "data-annotation-trainval-v1_1.json")

    if not os.path.exists(metadata_path):
        log.info(f"Downloading HoloAssist metadata from {metadata_url}...")
        with tqdm(unit="B", unit_scale=True, desc="Metadata", ncols=80) as t:
            urllib.request.urlretrieve(
                metadata_url, metadata_path, reporthook=tqdm_hook(t)
            )

    # Download and extract videos
    video_url = "https://hl2data.z5.web.core.windows.net/holoassist-data-release/video_pitch_shifted.tar"
    video_tar_path = os.path.join(raw_dataset_path, "video_pitch_shifted.tar")
    video_extract_path = os.path.join(raw_dataset_path, "videos")

    if not os.path.exists(video_extract_path):
        log.info(f"Downloading HoloAssist video data from {video_url}...")
        with tqdm(unit="B", unit_scale=True, desc="Video Data", ncols=80) as t:
            urllib.request.urlretrieve(
                video_url, video_tar_path, reporthook=tqdm_hook(t)
            )

        log.info(f"Extracting video data to {video_extract_path}...")
        os.makedirs(video_extract_path, exist_ok=True)
        with tarfile.open(video_tar_path, "r") as tar:
            tar.extractall(path=video_extract_path)
        log.info("Extraction complete.")

    return metadata_path, video_extract_path


def load_holoassist_data(data_dir: str, dataset: str) -> dict[str, dict]:
    metadata_path, _ = download_holoassist_data(data_dir, dataset)
    with open(metadata_path) as f:
        data = json.load(f)

    video_action_map = {}
    base_video_path = os.path.join(data_dir, "raw", dataset, "videos")

    for entry in data:
        video_name = entry["video_name"]
        fine_actions = [
            event
            for event in entry.get("events", [])
            if event.get("label") == "Fine grained action"
        ]
        if not fine_actions:
            continue

        export_py_dir = os.path.join(base_video_path, video_name, "Export_py")
        mp4_files = glob.glob(os.path.join(export_py_dir, "*.mp4"))
        video_file_path = mp4_files[0] if mp4_files else None

        video_action_map[video_name] = {
            "actions": fine_actions,
            "video_path": video_file_path,
        }

    return video_action_map


def get_holoassist_clip_info(
    clip_name: str, video_action_map: dict[str, list[dict]]
) -> tuple[str | None, float | None, float | None]:
    match = re.search(r"^(.*?)_coarse_(\d+)_fine_(\d+)", clip_name)
    if not match:
        log.info("Pattern not matched.")
        return None, None, None

    prefix = match.group(1)
    fine_id = int(match.group(3))

    if prefix not in video_action_map:
        log.info(f"Prefix '{prefix}' not found in video_action_map.")
        return None, None, None

    action = next(
        (d for d in video_action_map[prefix]["actions"] if d["id"] == fine_id), None
    )
    return prefix, action["start"], action["end"] if action else (None, None, None)


def get_agibot_clip_info(
    clip_name: str, data_dir: str
) -> tuple[str, str, str, int, int]:
    # Remove "clips/" prefix and ".mp4" suffix if present
    clip_name = clip_name.strip()
    if clip_name.startswith("clips/"):
        clip_name = clip_name[6:]
    if clip_name.endswith(".mp4"):
        clip_name = clip_name[:-4]

    # Split the clip name into components
    parts = clip_name.split("-")
    task_id = parts[0]
    episode_id = parts[1]
    video_name = parts[2]
    start_frame = int(parts[3])
    end_frame = int(parts[4])

    # Construct video path according to the new format
    video_path = (
        f"{data_dir}/agibot/observations/{task_id}/{episode_id}/videos/{video_name}.mp4"
    )

    return video_path, video_name, episode_id, start_frame, end_frame


def get_bridge_clip_info(clip_name: str) -> tuple[str, str, int, int]:
    folder_name, episode_id, _, clipframe, _ = clip_name.split(".")
    end_frame = int(clipframe[-3:])
    video_path = f"/nfs/kun2/users/homer/datasets/bridge_data_all/{folder_name.replace('-', '/')}/out.npy"
    return video_path, episode_id, 0, end_frame


def download_bridge_dataset(data_dir: str, split: str, dataset: str) -> None:
    target_dir = os.path.join(data_dir, "raw", dataset)
    os.makedirs(target_dir, exist_ok=True)

    # Define the reject pattern for the other split (train vs val)
    reject = "train" if split == "val" else "val"
    url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/"

    # Create destination directory to check for existing files
    bridge_dataset_dir = os.path.join(target_dir, "bridge_dataset", "1.0.0")
    os.makedirs(bridge_dataset_dir, exist_ok=True)

    log.info(f"Downloading Bridge dataset (split: {split}) to {target_dir}...")

    # Change to the target directory
    os.chdir(target_dir)

    # Use wget with more precise control to avoid downloading parent directory files
    subprocess.run(
        [
            "wget",
            "-r",
            "-nH",
            "--cut-dirs=4",
            "--no-clobber",
            "--no-parent",
            "--level=2",  # Prevent going up directories and limit depth
            f"--reject=index.html*,bridge_dataset-{reject}*",
            url,
        ],
        check=True,
    )

    # Report on downloaded files
    log.info(f"Finished downloading Bridge dataset (split: {split}).")


def save_clip(
    images: list[np.ndarray],
    data_dir: str,
    dataset: str,
    clip_name: str,
    task: str = "benchmark",
) -> None:
    clips_dir = os.path.join(data_dir, task, dataset, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    clip_path = os.path.join(clips_dir, clip_name)

    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(clip_path, fourcc, 30.0, (width, height))

    for img in images:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)

    out.release()
    log.info(f"Saved clip to {clip_path}")


def get_video_fps(video_path: str) -> float:
    """Get FPS using ffmpeg, fallback to 30.0 if failed."""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        if video_stream:
            fps_str = video_stream["r_frame_rate"]
            # Handle fractional frame rates like "30000/1001"
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                fps = num / den
            else:
                fps = float(fps_str)
            return fps if fps > 0 else 30.0
    except Exception:
        pass
    return 30.0


def preprocess_clip(
    clip_names: list[str],
    dataset: str,
    data_dir: str | None = None,
    split: str = "val",
    hf_token: str | None = None,
    task: str = "benchmark",
) -> None:
    if dataset == "holoassist":
        video_action_map = load_holoassist_data(data_dir, dataset)
    elif dataset == "bridgev2":
        download_bridge_dataset(data_dir, split, dataset)
    elif dataset == "agibot":
        download_agibot_data(data_dir, hf_token)
    else:
        raise ValueError(
            "Invalid dataset specified. Choose 'holoassist', 'bridge', or 'agibot'."
        )

    video_clip_map = {}

    for clip_name in clip_names:
        if dataset == "holoassist":
            video_name, start, end = get_holoassist_clip_info(
                clip_name, video_action_map
            )
            if video_name:
                video_clip_map.setdefault(video_name, []).append(
                    {
                        "clip_name": clip_name,
                        "start_frame": start,
                        "end_frame": end,
                        "video_path": video_action_map[video_name]["video_path"],
                    }
                )
        elif dataset == "bridgev2":
            video_name, episode_id, start, end = get_bridge_clip_info(clip_name)
            video_clip_map.setdefault(video_name, []).append(
                {
                    "clip_name": clip_name,
                    "episode_id": episode_id,
                    "start_frame": start,
                    "end_frame": end,
                }
            )
        elif dataset == "agibot":
            video_path, video_name, episode_id, start, end = get_agibot_clip_info(
                clip_name, f"{data_dir}/raw"
            )
            video_clip_map.setdefault(video_name, []).append(
                {
                    "clip_name": clip_name,
                    "episode_id": episode_id,
                    "start_frame": start,
                    "end_frame": end,
                    "video_path": video_path,
                }
            )
        else:
            raise ValueError(
                "Invalid dataset specified. Choose 'holoassist', 'bridgev2', or 'agibot'."
            )

    if dataset == "bridgev2":
        ds = tfds.load(
            "bridge_dataset", split=split, data_dir=f"{data_dir}/raw/{dataset}"
        )
        log.info(f"Number of videos in Bridge V2 {split} split: {len(ds)}")
        n_no_instruction = 0
        try:
            for episode in ds:
                if not episode["episode_metadata"]["has_language"].numpy():
                    n_no_instruction += 1
                    continue

                video_name_tfds = (
                    episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
                )
                episode_id_tfds = str(episode["episode_metadata"]["episode_id"].numpy())

                for clip_info in video_clip_map.get(video_name_tfds, []):
                    if episode_id_tfds == clip_info["episode_id"]:
                        images = [
                            step["observation"]["image_0"].numpy()
                            for step in episode["steps"]
                        ]
                        start_frame = clip_info["start_frame"]
                        end_frame = clip_info["end_frame"]
                        images_clipped = images[start_frame : end_frame + 1]
                        save_clip(
                            images_clipped, data_dir, dataset, clip_info["clip_name"]
                        )
        except Exception:
            log.warning("Clip processing error!")

        log.info(f"Number of episodes without instructions: {n_no_instruction}")

    elif dataset == "holoassist" or dataset == "agibot":
        log.info(f"Number of videos in {dataset}: {len(video_clip_map)}")

        for video_id, clips in video_clip_map.items():  # noqa: B007
            for clip in clips:
                input_video_path = clip["video_path"]

                # Skip processing if video file doesn't exist
                if not os.path.exists(input_video_path):
                    log.warning(
                        f"Skipping clip {clip['clip_name']} - video file not found"
                    )
                    continue

                # Calculate start and end times in seconds
                if dataset == "agibot":
                    # Get actual FPS using OpenCV
                    fps = get_video_fps(input_video_path)
                    start_time = clip["start_frame"] / fps
                    end_time = clip["end_frame"] / fps
                elif dataset == "holoassist":
                    # For holoassist, start_frame and end_frame are already in seconds
                    start_time = clip["start_frame"]
                    end_time = clip["end_frame"]

                # Create output path using clip name
                output_video = os.path.join(
                    data_dir, task, dataset, "clips", f"{clip['clip_name']}"
                )
                os.makedirs(os.path.dirname(output_video), exist_ok=True)

                try:
                    # Force software decoding for AV1 compatibility
                    ffmpeg.input(
                        input_video_path, ss=start_time, to=end_time, hwaccel="none"
                    ).output(
                        output_video,
                        vcodec="libx264",
                        acodec="aac",
                        **{"c:v": "libx264", "preset": "fast"},
                    ).run(quiet=True, overwrite_output=True)
                    log.info(f"Saved clip to: {output_video}")
                except Exception as e:
                    log.warning(f"Failed to extract clip {clip['clip_name']}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="*", choices=ALL_DATASETS)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--token", type=str)
    parser.add_argument(
        "--task", type=str, choices=["benchmark", "sft", "rl"], required=True
    )

    args = parser.parse_args()

    hf_dataset_map = {
        "benchmark": "nvidia/Cosmos-Reason1-Benchmark",
        "sft": "nvidia/Cosmos-Reason1-SFT-Dataset",
        "rl": "nvidia/Cosmos-Reason1-RL-Dataset",
    }

    split = "val" if args.task == "benchmark" else "train"
    hf_dataset = hf_dataset_map[args.task]
    hf_token = args.token or os.environ.get("HF_TOKEN")

    if not hf_token:
        log.warning(
            "No Hugging Face token (HF_TOKEN) provided via args or environment."
        )

    datasets = args.datasets or ALL_DATASETS
    for dataset in datasets:
        ds = load_dataset(hf_dataset, dataset)

        clip_names = []
        for dataset_name in ds.keys():
            clip_names.extend(
                [item.split("clips/")[-1] for item in ds[dataset_name]["video"]]
            )

        preprocess_clip(
            clip_names=clip_names,
            dataset=dataset,
            data_dir=args.data_dir,
            split=split,
            hf_token=hf_token,
            task=args.task,
        )


if __name__ == "__main__":
    main()
