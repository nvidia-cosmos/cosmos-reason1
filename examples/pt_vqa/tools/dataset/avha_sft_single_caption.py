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

"""Supervised Fine-Tuning (SFT) dataset with plain text caption loading."""
# ruff: noqa: E402

from cosmos_reason1_utils.script import init_script

init_script()

import argparse
import copy
import json
import os
from pathlib import Path

import toml
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.util import basename_from_modelpath
from torch.utils.data import Dataset
from transformers import AutoTokenizer

FPS = 1
MAX_PIXELS = 81920


class CosmosSFTDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.videos_dir = self.dataset_path / "videos"
        self.captions_dir = self.dataset_path / "metas"
        
        # Load all available video files and create dataset entries
        self.data_entries = self._load_dataset_entries()

    def _load_dataset_entries(self):
        """Load dataset entries by scanning video files and matching with JSON captions"""
        entries = []
        
        # Validate directories exist
        for dir_path, dir_name in [(self.videos_dir, "videos"), (self.captions_dir, "metas")]:
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory {dir_path} does not exist. Please check the dataset path.")
        
        # Scan video files with the specific naming pattern
        video_files = list(self.videos_dir.glob("*.camera_front_wide_120fov.mp4"))
        print(f"Found {len(video_files)} video files")
        
        for video_file in video_files:
            # Extract base filename (remove .camera_front_wide_120fov.mp4 suffix)
            video_filename = video_file.name
            if video_filename.endswith(".camera_front_wide_120fov.mp4"):
                base_id = video_filename[:-len(".camera_front_wide_120fov.mp4")]
            else:
                print(f"Warning: Unexpected video filename format: {video_filename}, skipping")
                continue
            
            # Look for corresponding caption file
            caption_file = self.captions_dir / f"{base_id}.label.json"
            
            # Check if caption file exists
            if not caption_file.exists():
                print(f"Warning: Caption file not found for {base_id}, skipping")
                continue
                
            # Load caption data from JSON
            try:
                with open(caption_file, 'r') as f:
                    caption_json = json.load(f)
                    caption_data = str(caption_json)
                        
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load caption for {base_id}: {e}, skipping")
                continue
            
            entries.append({
                "video_id": base_id,
                "video_path": str(video_file),
                "caption_data": caption_data
            })
        
        print(f"Successfully loaded {len(entries)} dataset entries")
        return entries
    
    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        self.config = config
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx: int):
        """
        Return conversations in the expected format
        """
        try:
            entry = self.data_entries[idx]
            caption_data = entry["caption_data"]
        except Exception as e:
            print(f"Error accessing entry {idx}: {e}")
            raise
        
        # Use the entire caption file content as the description
        full_description = caption_data if caption_data.strip() else "Describe what you see in this video."
        
        # Create conversations in the expected format
        conversations = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": entry["video_path"],
                        "max_pixels": MAX_PIXELS,
                        "fps": FPS,
                    },
                    {
                        "type": "text",
                        "text": "Describe what you see in this video.",
                    },
                ]
            },
            {
                "role": "assistant",
                "content": full_description
            }
        ]

        return conversations
    
    
    def get_video_path(self, idx: int) -> str:
        """Get video path for a specific sample"""
        entry = self.data_entries[idx]
        return entry["video_path"]
    
    def get_video_id(self, idx: int) -> str:
        """Get video ID for a specific sample"""
        entry = self.data_entries[idx]
        return entry["video_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config) as f:
        config = toml.load(f)
    config = Config.from_dict(config)

    def get_dataset(config: CosmosConfig) -> Dataset:
        # Get dataset path from config, with fallback to default path
        dataset_config = config.train.train_policy.dataset
        if hasattr(dataset_config, 'path'):
            dataset_path = dataset_config.path
        elif hasattr(dataset_config, 'name') and dataset_config.name.startswith('/'):
            # If name is a path, use it directly
            dataset_path = dataset_config.name
        else:
            # Default fallback path
            dataset_path = '/project/cosmos/jingyij/dataset/av_human_annotated/v0_1/train'
        
        print(f"Loading local dataset from: {dataset_path}")
        return CosmosSFTDataset(dataset_path)

    # It is best practice to pass the dataset as a factory function
    # so that the dataset can be loaded on demand. (Not all workers need it)
    launch_worker(
        dataset=get_dataset,
    )

