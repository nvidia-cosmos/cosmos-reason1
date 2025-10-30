#!/usr/bin/env python3
"""
Diagnostic script to identify problematic files in the dataset.
"""

import json
import sys
from pathlib import Path

def diagnose_dataset(dataset_path="/project/cosmos/jingyij/dataset/nexar/DESC_caption/v2/train"):
    """Diagnose potential issues in the dataset"""
    dataset_path = Path(dataset_path)
    captions_dir = dataset_path / "metas_cleaned"
    
    print(f"Diagnosing dataset at: {dataset_path}")
    
    problematic_files = []
    total_files = 0
    
    for caption_file in captions_dir.glob("*.txt"):
        total_files += 1
        video_id = caption_file.stem
        
        try:
            with open(caption_file, 'r') as f:
                caption_data = json.load(f)
            
            # Check for problematic notice structures
            if isinstance(caption_data, list):
                for item in caption_data:
                    if isinstance(item, dict) and "notice" in item:
                        notice_data = item["notice"]
                        if isinstance(notice_data, list):
                            for i, notice in enumerate(notice_data):
                                if not isinstance(notice, str):
                                    problematic_files.append({
                                        "file": str(caption_file),
                                        "video_id": video_id,
                                        "issue": f"Notice item {i} is not a string: {type(notice)} - {notice}"
                                    })
            elif isinstance(caption_data, dict) and "notice" in caption_data:
                notice_data = caption_data["notice"]
                if isinstance(notice_data, list):
                    for i, notice in enumerate(notice_data):
                        if not isinstance(notice, str):
                            problematic_files.append({
                                "file": str(caption_file),
                                "video_id": video_id,
                                "issue": f"Notice item {i} is not a string: {type(notice)} - {notice}"
                            })
                            
        except Exception as e:
            problematic_files.append({
                "file": str(caption_file),
                "video_id": video_id,
                "issue": f"Failed to parse JSON: {e}"
            })
    
    print(f"Processed {total_files} files")
    print(f"Found {len(problematic_files)} problematic files")
    
    if problematic_files:
        print("\nProblematic files:")
        for issue in problematic_files[:10]:  # Show first 10
            print(f"  {issue['video_id']}: {issue['issue']}")
        
        if len(problematic_files) > 10:
            print(f"  ... and {len(problematic_files) - 10} more")
    else:
        print("No problematic files found!")
    
    return problematic_files

if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "/project/cosmos/jingyij/dataset/nexar/DESC_caption/v2/train"
    issues = diagnose_dataset(dataset_path)
    sys.exit(1 if issues else 0)
