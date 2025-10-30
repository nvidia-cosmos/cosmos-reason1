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
from typing import Dict, List, Any, Optional

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
    def __init__(self, dataset_path: str, schema_file: Optional[str] = None, validate_schema: bool = True):
        self.dataset_path = Path(dataset_path)
        self.videos_dir = self.dataset_path / "videos"
        self.captions_dir = self.dataset_path / "metas"
        self.validate_schema = validate_schema
        
        # Load schema for validation if provided
        self.schema = self._load_schema(schema_file) if schema_file else self._get_default_schema()
        
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
                    
                    # Normalize data to handle common inconsistencies
                    caption_json = self._normalize_json_data(caption_json)
                    
                    # Validate JSON structure using schema
                    if self._validate_json_structure(caption_json, base_id):
                        caption_data = str(caption_json)
                    else:
                        print(f"Warning: Invalid JSON structure for {base_id}, skipping")
                        continue
                        
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
    
    def _load_schema(self, schema_file: str) -> Dict[str, Any]:
        """Load schema from JSON file"""
        try:
            with open(schema_file, 'r') as f:
                content = f.read()
                # Handle the format "JSON Template (schema with possible values):"
                if content.startswith("JSON Template"):
                    lines = content.split('\n')
                    json_start = next(i for i, line in enumerate(lines) if line.strip().startswith('{'))
                    json_content = '\n'.join(lines[json_start:])
                    schema = json.loads(json_content)
                else:
                    schema = json.loads(content)
                
                # Normalize Unicode characters in schema values
                def normalize_schema_values(obj):
                    if isinstance(obj, str):
                        return obj.replace('‑', '-').replace('—', '-')
                    elif isinstance(obj, list):
                        return [normalize_schema_values(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {k: normalize_schema_values(v) for k, v in obj.items()}
                    return obj
                
                schema = normalize_schema_values(schema)
                
                # Add required_fields if not present
                if "required_fields" not in schema:
                    schema["required_fields"] = {
                        "weather", "lighting", "road_conditions", "traffic_light", "traffic_sign", 
                        "road_type", "junction_type", "lane_type", "safety_analysis", 
                        "driving_difficulty", "rule_violation", "interesting_scenario", "critical_object"
                    }
                
                return schema
                
        except Exception as e:
            print(f"Warning: Could not load schema from {schema_file}: {e}")
            return self._get_default_schema()
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Default schema based on your JSON template"""
        return {
            "weather": ["Clear skies", "Overcast", "Partly cloudy", "Light rain", "Heavy rain", "Light snow", "Dense fog", "N/A"],
            "lighting": [
                "Daytime (diffused light)", 
                "Daytime (low-contrast daylight)", 
                "Daytime (direct sunlight)", 
                "Night (poorly lit; few/no streetlights)", 
                "Night (well-lit by streetlights)"
            ],
            "driving_difficulty": [1, 2, 3, 4],
            "rule_violation": ["False", "True"],
            "interesting_scenario": ["True", "False"],
            "required_fields": {
                "weather", "lighting", "road_conditions", "traffic_light", "traffic_sign", 
                "road_type", "junction_type", "lane_type", "safety_analysis", 
                "driving_difficulty", "rule_violation", "interesting_scenario", "critical_object"
            }
        }
    
    def _validate_json_structure(self, json_data: dict, video_id: str) -> bool:
        """Validate JSON structure using loaded schema"""
        if not self.validate_schema:
            return True
            
        validation_errors = []
        
        # Check required fields
        required_fields = self.schema.get("required_fields", set())
        missing_fields = required_fields - set(json_data.keys())
        if missing_fields:
            validation_errors.append(f"Missing fields: {missing_fields}")
        
        # Validate enum fields
        enum_validations = [
            ("weather", self.schema.get("weather", [])),
            ("lighting", self.schema.get("lighting", [])),
            ("driving_difficulty", self.schema.get("driving_difficulty", [])),
            ("rule_violation", self.schema.get("rule_violation", [])),
            ("interesting_scenario", self.schema.get("interesting_scenario", []))
        ]
        
        for field_name, valid_values in enum_validations:
            if field_name in json_data:
                value = json_data[field_name]
                # Handle string/bool conversion for rule_violation and interesting_scenario
                if field_name in ["rule_violation", "interesting_scenario"]:
                    if isinstance(value, bool):
                        value = str(value)
                
                if value not in valid_values:
                    validation_errors.append(f"{field_name} has invalid value '{value}', expected one of {valid_values}")
        
        # Validate critical_object structure
        if "critical_object" in json_data:
            critical_objects = json_data["critical_object"]
            if isinstance(critical_objects, list):
                for i, obj in enumerate(critical_objects):
                    if not isinstance(obj, dict):
                        validation_errors.append(f"critical_object[{i}] is not a dictionary")
                        continue
                    
                    required_obj_fields = {"box", "object_type", "critical_reasoning"}
                    missing_obj_fields = required_obj_fields - set(obj.keys())
                    if missing_obj_fields:
                        validation_errors.append(f"critical_object[{i}] missing fields: {missing_obj_fields}")
                    
                    # Validate box structure
                    if "box" in obj and isinstance(obj["box"], dict):
                        required_box_fields = {"x1", "x2", "y1", "y2"}
                        missing_box_fields = required_box_fields - set(obj["box"].keys())
                        if missing_box_fields:
                            validation_errors.append(f"critical_object[{i}].box missing fields: {missing_box_fields}")
        
        # Report validation results
        if validation_errors:
            print(f"Validation errors for {video_id}:")
            for error in validation_errors:
                print(f"  - {error}")
            return False
            
        return True
    
    def _normalize_json_data(self, json_data: dict) -> dict:
        """Normalize common inconsistencies in JSON data"""
        normalized = json_data.copy()
        
        # Normalize Unicode characters (en-dash to regular hyphen)
        def normalize_unicode_text(text, preserve_null=False):
            if text is None:
                return None if preserve_null else ""
            if isinstance(text, str):
                # Replace en-dash (‑) with regular hyphen (-)
                text = text.replace('‑', '-')
                # Replace em-dash (—) with regular hyphen (-)
                text = text.replace('—', '-')
                # Replace other common Unicode variations
                text = text.replace(''', "'").replace(''', "'")
                text = text.replace('"', '"').replace('"', '"')
                return text
            return str(text)  # Convert other types to string
        
        # Apply Unicode normalization to required text fields
        required_text_fields = ["weather", "lighting", "road_conditions", "traffic_light", "traffic_sign", 
                               "road_type", "junction_type", "lane_type", "safety_analysis"]
        
        for field in required_text_fields:
            if field in normalized:
                normalized[field] = normalize_unicode_text(normalized[field], preserve_null=False)
        
        # Apply Unicode normalization to nullable text fields
        nullable_text_fields = ["additional_traffic_rules", "additional_map_rules"]
        
        for field in nullable_text_fields:
            if field in normalized:
                normalized[field] = normalize_unicode_text(normalized[field], preserve_null=True)
        
        # Normalize traffic_light field (handle "NA", "N/A", "No traffic light" variations)
        if "traffic_light" in normalized:
            traffic_light = normalized["traffic_light"]
            if isinstance(traffic_light, str):
                traffic_light = traffic_light.strip()
                if traffic_light.lower() in ["na", "n/a", "no traffic light", "no traffic lights", "none", ""]:
                    normalized["traffic_light"] = "No traffic light"
        
        # Normalize boolean fields
        for bool_field in ["rule_violation", "interesting_scenario"]:
            if bool_field in normalized:
                value = normalized[bool_field]
                if isinstance(value, str):
                    normalized[bool_field] = value.lower() == "true"
                elif isinstance(value, bool):
                    normalized[bool_field] = value
        
        # Ensure driving_difficulty is integer
        if "driving_difficulty" in normalized:
            try:
                normalized["driving_difficulty"] = int(normalized["driving_difficulty"])
            except (ValueError, TypeError):
                pass  # Let validation catch this
        
        # Normalize critical_object text fields
        if "critical_object" in normalized and isinstance(normalized["critical_object"], list):
            for obj in normalized["critical_object"]:
                if isinstance(obj, dict):
                    for text_field in ["object_type", "critical_reasoning"]:
                        if text_field in obj:
                            obj[text_field] = normalize_unicode_text(obj[text_field], preserve_null=False)
        
        return normalized
    
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
        
        # Check for schema file in config or use default location
        schema_file = None
        validate_schema = True
        
        if hasattr(dataset_config, 'schema_file'):
            schema_file = dataset_config.schema_file
        if hasattr(dataset_config, 'validate_schema'):
            validate_schema = dataset_config.validate_schema
            
        # Try to find schema file in common locations
        if not schema_file:
            potential_schema_files = [
                os.path.join(dataset_path, 'json_schema.txt'),
                os.path.join(os.path.dirname(dataset_path), 'json_schema.txt'),
                'json_schema.txt'
            ]
            for potential_file in potential_schema_files:
                if os.path.exists(potential_file):
                    schema_file = potential_file
                    break
        
        print(f"Loading local dataset from: {dataset_path}")
        if schema_file:
            print(f"Using schema file: {schema_file}")
        
        return CosmosSFTDataset(dataset_path, schema_file=schema_file, validate_schema=validate_schema)

    # It is best practice to pass the dataset as a factory function
    # so that the dataset can be loaded on demand. (Not all workers need it)
    launch_worker(
        dataset=get_dataset,
    )

