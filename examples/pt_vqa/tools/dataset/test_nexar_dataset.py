#!/usr/bin/env python3
"""
Simple test script to verify the modified nexar_sft.py dataset loading works correctly.
"""

import sys
import os
import json
import pickle
from pathlib import Path

# Minimal implementation to test dataset loading without full cosmos dependencies
class CosmosSFTDataset:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.videos_dir = self.dataset_path / "videos"
        self.captions_dir = self.dataset_path / "metas_cleaned"
        self.t5_dir = self.dataset_path / "t5_xxl"
        
        # Load all available video files and create dataset entries
        self.data_entries = self._load_dataset_entries()
    
    def _load_dataset_entries(self):
        """Load dataset entries by scanning video files and matching with captions and embeddings"""
        entries = []
        
        # Validate directories exist
        for dir_path, dir_name in [(self.videos_dir, "videos"), (self.captions_dir, "metas_cleaned")]:
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory {dir_path} does not exist. Please check the dataset path.")
        
        # Scan video files
        video_files = list(self.videos_dir.glob("*.mp4"))
        print(f"Found {len(video_files)} video files")
        
        for video_file in video_files:
            video_id = video_file.stem  # Get filename without extension
            caption_file = self.captions_dir / f"{video_id}.txt"
            t5_file = self.t5_dir / f"{video_id}.pickle"
            
            # Check if caption file exists
            if not caption_file.exists():
                print(f"Warning: Caption file not found for {video_id}, skipping")
                continue
                
            # Load caption data
            try:
                with open(caption_file, 'r') as f:
                    caption_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not load caption for {video_id}: {e}, skipping")
                continue
            
            # Load T5 embedding if available
            t5_embedding = None
            if t5_file.exists():
                try:
                    with open(t5_file, 'rb') as f:
                        t5_embedding = pickle.load(f)
                except Exception as e:
                    print(f"Warning: Could not load T5 embedding for {video_id}: {e}")
            
            entries.append({
                "video_id": video_id,
                "video_path": str(video_file),
                "caption_data": caption_data,
                "t5_embedding": t5_embedding
            })
        
        print(f"Successfully loaded {len(entries)} dataset entries")
        return entries
    
    def __len__(self):
        return len(self.data_entries)
    
    def get_t5_embedding(self, idx: int):
        """Get T5 embedding for a specific sample"""
        entry = self.data_entries[idx]
        return entry.get("t5_embedding")
    
    def get_video_path(self, idx: int) -> str:
        """Get video path for a specific sample"""
        entry = self.data_entries[idx]
        return entry["video_path"]
    
    def get_video_id(self, idx: int) -> str:
        """Get video ID for a specific sample"""
        entry = self.data_entries[idx]
        return entry["video_id"]
    
    def __getitem__(self, idx: int):
        """Return conversations in the expected format"""
        entry = self.data_entries[idx]
        caption_data = entry["caption_data"]
        
        # Extract description from caption data
        description = ""
        driving_difficulty = ""
        notices = []
        
        if isinstance(caption_data, list) and len(caption_data) > 0:
            # Handle the JSON format from your dataset
            for item in caption_data:
                if isinstance(item, dict):
                    if "description" in item:
                        description = item["description"]
                    elif "driving difficulity explanation" in item:
                        driving_difficulty = item["driving difficulity explanation"]
                    elif "notice" in item:
                        notice_data = item["notice"]
                        if isinstance(notice_data, list):
                            # Filter to ensure all items are strings
                            notices = [str(n) for n in notice_data if n is not None]
                        elif isinstance(notice_data, str):
                            notices = [notice_data]
        elif isinstance(caption_data, dict):
            description = caption_data.get("description", "")
            driving_difficulty = caption_data.get("driving difficulity explanation", "")
            notice_data = caption_data.get("notice", [])
            if isinstance(notice_data, list):
                notices = [str(n) for n in notice_data if n is not None]
            elif isinstance(notice_data, str):
                notices = [notice_data]
        
        # Combine all information into a comprehensive description
        full_description = description
        if driving_difficulty:
            full_description += f" {driving_difficulty}"
        if notices:
            # Safely join notices, ensuring they are all strings
            notice_text = " ".join(notices)
            if notice_text.strip():
                full_description += f" {notice_text}"
        
        if not full_description.strip():
            full_description = "Describe what you see in this video."
        
        # Create conversations in the expected format
        conversations = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": entry["video_path"],
                        "max_pixels": 81920,
                        "fps": 1,
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

def test_dataset_loading():
    """Test loading the local dataset"""
    dataset_path = "/project/cosmos/jingyij/dataset/nexar/DESC_caption/v2/train"
    
    print(f"Testing dataset loading from: {dataset_path}")
    
    try:
        # Create dataset instance
        dataset = CosmosSFTDataset(dataset_path)
        
        print(f"Dataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test first sample
            print("\nTesting first sample:")
            sample = dataset[0]
            print(f"Sample type: {type(sample)}")
            print(f"Sample keys: {sample[0].keys() if isinstance(sample, list) and len(sample) > 0 else 'N/A'}")
            
            # Test additional methods
            video_path = dataset.get_video_path(0)
            video_id = dataset.get_video_id(0)
            t5_embedding = dataset.get_t5_embedding(0)
            
            print(f"Video ID: {video_id}")
            print(f"Video path: {video_path}")
            print(f"T5 embedding available: {t5_embedding is not None}")
            if t5_embedding is not None:
                print(f"T5 embedding type: {type(t5_embedding)}")
            
            # Print conversation structure
            print("\nConversation structure:")
            for i, conv in enumerate(sample):
                print(f"  Message {i}: role={conv['role']}")
                if conv['role'] == 'user':
                    content = conv['content']
                    print(f"    Content items: {len(content)}")
                    for j, item in enumerate(content):
                        print(f"      Item {j}: type={item['type']}")
                        if item['type'] == 'video':
                            print(f"        Video path: {item['video']}")
                        elif item['type'] == 'text':
                            print(f"        Text: {item['text'][:100]}...")
                else:
                    print(f"    Content: {conv['content'][:200]}...")
        
        print("\nDataset test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    sys.exit(0 if success else 1)
