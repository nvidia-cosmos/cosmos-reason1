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
        Return conversations in the expected format with complex driving annotation prompts
        """
        try:
            entry = self.data_entries[idx]
            caption_data = entry["caption_data"]
        except Exception as e:
            print(f"Error accessing entry {idx}: {e}")
            raise
        
        # Use the entire caption file content as the description
        full_description = caption_data if caption_data.strip() else "{}"
        
        # System prompt for expert video annotation
        system_prompt = """You are an expert video annotator, and your job is to produce high-quality annotations for driving videos. The annotations will be used as training data for autonomous vehicles, so they must be both detailed and accurate. You should pay particular attention to any details in the video that are relevant to driving safety.

Carefully review the following video clip, and provide annotations in JSON format, according to the rubric given by the user."""

        # Detailed user prompt with JSON schema
        user_prompt = """Please analyze the video, and provide accurate and detailed annotations. Your answer must be provided in the following JSON format:

{
  "weather": <string>, 
      Describe the weather conditions. Should be one of the following:

      "Clear skies",
      "Overcast",
      "Partly cloudy",
      "Light rain",
      "Heavy rain",
      "Light snow",
      "Dense fog",
      "N/A"

  "lighting": <string>,
      Describe the lighting conditions. Should be one of the following:

      "Daytime (diffused light)",
      "Daytime (low-contrast daylight)",
      "Daytime (direct sunlight)",
      "Night (poorly lit; few/no streetlights)",
      "Night (well-lit by streetlights)"     

  "road_conditions": <string>, 
      Provide a brief description of both the condition of the road (dry, wet, icy etc.), and the 
      type of road (asphalt, cobblestones, gravel, narrow, etc.). For example:

      "Dry, clean asphalt"
      "Wet asphalt with puddles"
      "Wet asphalt with patches of ice"
      "Icy and snow-covered"
      "Dry, clean asphalt with snow on the sides of the road"
      "A wet cobblestone road"
      "Narrow road with mud and gravel"
      "A dry dirt road"
      ... etc.

  "traffic_light": <string>,
      Describe any traffic lights, and the ego vehicle's response to them. For example:
   
      "No traffic light"
      "The traffic light is green, so the ego vehicle maintains speed and proceeds through the intersection."
      "The light is red, so the ego vehicle to comes to a full stop."
      "The signal is flashing yellow, instructing the ego vehicle to proceed with caution."
      "The traffic light turns from green to yellow and then red. The ego vehicle slows and stops."
      "The traffic light turns yellow, and the ego vehicle acclerates through it before it changes to red."
      "A left turn arrow goes from red to green; the ego vehicle enters the intersection and turns left."
      "The traffic light is not operational. The ego vehicle stops, then proceeds."
      ... etc. 
      
  "traffic_sign": <string>,
      Describe any relevant traffic signs that the ego vehicle must pay attention to. If there 
      are multiple signs, mention all of them. For example:

      "There is a stop sign marking a four-way stop."
      "A yield sign requires the ego vehicle to yield to cross traffic."
      "Speed limit 30mph"
      "One-way (pointing right)"
      "School zone, speed limit 15mph"
      "No U-turn sign."
      "Road work ahead."
      "A 'Do Not Enter' sign marks the end of a freeway off-ramp."
      ... etc.

  "additional_traffic_rules": <string> or null,
      Provide one or two sentences that describe any special traffic rules that apply on this
      particular section of road. For example:

      "There is a construction zone ahead, and cars must yield to construction."
      "There is a school zone, with a speed limit of 15mph."
      "There is a pedestrian crosswalk." 
      "End of no parking zone."
      "Two left lanes must turn left, right lane must turn right."
      ... etc.

  "road_type": <string>, 
      Describe the kind of road that the ego vehicle is traveling on. For example:

      "Local street -- low-speed mixed residential/commerical"
      "Collector road -- moderate speed with frequent driveways and intersections"
      "Arterial road -- high-capacity through traffic, limited direct access"
      "Highway/freeway"
      "Freeway on-ramp"
      "Country road"
      "Narrow urban alleyway"
      ... etc.

  "junction_type": <string>,
      Describe any junctions or intersections that the ego vehicle either passes through, or
      is forced to stop at. For example: 

      "Straight segment - uninterrupted roadway with no intersections or turns"
      "Four-way intersection -- crossing of two roads with traffic controls"
      "T-intersection without traffic controls"
      "Curved road segment turning left, no intersections"
      "Large roundabout"
      ... etc.  

  "lane_type": <string>,
      Describe the number and type of lanes in the road that the ego vehicle is driving on.  
      For example:

      "Single lane each direction, no turn lanes"
      "Two lanes each direction, plus a center turn lane"
      "Four lanes in the ego's direction, including two right-turn-only lanes and a left-turn only lane."
      "Two lanes, one way street"
      "There is a single lane in each direction, plus a bicycle lane"
      ... etc.

  "additional_map_rules": <string> or null,
      Provide one or two sentences that describe any markings on the road itself that indicate 
      special traffic rules for this particular section of the road. For example:

      "There are pedestrian crosswalk markings on the road." 
      "Bicycle lane markings are visible on right."
      "The road is shared with the tram lines."
      "There's an HOV carpool lane on the left with HOV diamonds and the letters 'CAR POOL LANE'."
      "A right-turn-only arrow in the rightmost lane."
      ... etc.

  "interactive_expanded_metaaction": null or [ <list of strings> ],
      If the ego vehicle's behavior falls into any of the following categories, provide a list of 
      which categories best describe the behavior. Return null if no categories match. Categories 
      must be one of:

      "In-lane nudging",
      "Yield to VRUs",
      "Vehicle following",
      "Out-of-lane nudging",
      "Yield to vehicles",
      "Overtake VRUs",
      "Vehicle cut-in ahead",
      "React to animals",
      "Overtake vehicles"
  
  "safety_analysis": <string>,
      Write a short paragraph describing the situation that the ego vehicle is driving in, and the
      sequence of actions that the vehicle performs (e.g. lane changes, turns, stops, etc.) The 
      paragraph should focus on describing any events or part of the situation that are relevant 
      to safety, e.g. the presence of pedestrians or parked cars, sudden lane changes by the ego 
      vehicle or other drivers, difficult driving conditions, and so on.

  "driving_difficulty": <integer between 1 and 4>,
      On a scale of 1 to 4, rate the driving difficulty.  

      1: very easy -- clear weather, low traffic, no hazards.
      2: normal -- some traffic or adverse weather, but not too much.
      3: difficult -- heavy traffic, rain, pedestrians, etc.
      4: very diffult -- poor visibility, rush-hour traffic, treacherous road conditions, etc.

  "rule_violation": <boolean>,
      "True" if there was a violation of traffic law in the video, "False" otherwise.

  "interesting_scenario": <boolean>,
      "True" if this driving scenario contains interesting or unusual characterists that make 
      it good training data. "False" if it is normal or uninteresting driving conditions.

  "critical_object": [ <list of objects> ],
      Provide a list of any critical objects that you can see in the video. A critical object is
      an object that the driver must pay attention and respond to in order to drive safely.  
      Examples include other vehicles, pedestrians, animals, or traffic hazards. Each critical 
      object should have a bounding box with coordinates in pixels, a description of the object 
      type, and a description of why the object is considered to be to be critical. Each critical 
      object has the following format:

      { 
        "box": {
          "x1": <float>,
          "x2": <float>,
          "y1": <float>,
          "y2": <float>,
        },

        "object_type": <string>,
            Provide a short description of the object type, for example:

            "Vehicle -- Passenger car",
            "Vehicle -- Light trucks & SUVs",
            "Vehicle -- Heavy trucks & lorries",
            "Vehicle -- Bicycle",
            "Pedestrian",
            "SUV with attached trailer",
            "Animal -- Domestic dog",
            "Debris in the road",
            ... etc.

        "critical_reasoning": <string>,
            Provide one or two sentences describing why the object is considered to be critical,
            and how the ego vehicle responded in order to drive safely. For example:

            "A white SUV is immediately ahead of the ego, so the driver must maintain safe following distance.",
            "The car in front braked, so the ego vehicle slowed down.",
            "The ego vehicle is yielding to a pedestrian crossing the crosswalk.",
            "The ego vehicle swerves to avoid debris in the lane."
            "A dog jumped out in front of the ego vehicle, thus causing it to brake."
            "The ego vehicle nudged out of its lane to avoid a bicycle on the right."
            ... etc.
      }
}"""
        
        # Create conversations in the expected format with system and user prompts
        conversations = [
            {
                "role": "system",
                "content": system_prompt
            },
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
                        "text": user_prompt,
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

