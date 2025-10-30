# Nexar SFT Dataset Modifications

## Overview
The `nexar_sft.py` script has been modified to work with your local dataset structure instead of HuggingFace datasets.

## Dataset Structure Expected
```
/project/cosmos/jingyij/dataset/nexar/DESC_caption/v2/train/
├── videos/                 # Video files (.mp4)
├── metas_cleaned/         # Caption files (.txt, JSON format)
└── t5_xxl/               # T5 embeddings (.pickle files)
```

## Key Changes Made

### 1. Dataset Loading
- **Before**: Used HuggingFace `load_dataset()` 
- **After**: Scans local directories for video files and matches them with captions and embeddings

### 2. File Structure Handling
- Videos are loaded from `videos/` directory
- Captions are loaded from `metas_cleaned/` directory (JSON format)
- T5 embeddings are loaded from `t5_xxl/` directory (pickle format)
- Files are matched by UUID (filename without extension)

### 3. Caption Processing
The script now handles your JSON caption format:
```json
[
    {
        "description": "Main video description..."
    },
    {
        "driving difficulity explanation": "Difficulty explanation..."
    },
    {
        "notice": ["Notice 1", "Notice 2", ...]
    }
]
```

All three components are combined into a comprehensive description for training.

### 4. New Methods Added
- `get_t5_embedding(idx)`: Access T5 embeddings for a sample
- `get_video_path(idx)`: Get video file path for a sample  
- `get_video_id(idx)`: Get video UUID for a sample

### 5. Configuration
The dataset path can be configured in your config file:
```toml
[train.train_policy.dataset]
path = "/path/to/your/dataset/train"
# OR
name = "/path/to/your/dataset/train"  # if name starts with '/', treated as path
```

Default fallback: `/project/cosmos/jingyij/dataset/nexar/DESC_caption/v2/train`

## Usage

### Basic Usage
```python
from nexar_sft import CosmosSFTDataset

# Create dataset
dataset = CosmosSFTDataset("/path/to/dataset/train")

# Get sample
conversations = dataset[0]
video_path = dataset.get_video_path(0)
t5_embedding = dataset.get_t5_embedding(0)
```

### With Training Script
```bash
python nexar_sft.py --config your_config.toml
```

## Test Results
- ✅ Successfully loaded 1,918 dataset entries
- ✅ Video files properly matched with captions
- ✅ T5 embeddings loaded correctly (numpy arrays)
- ✅ Conversation format matches expected structure
- ✅ All caption components (description, difficulty, notices) combined

## Files Modified
1. `nexar_sft.py` - Main dataset class modifications
2. `test_nexar_dataset.py` - Test script to verify functionality

## Dependencies Added
- `json` - For parsing caption files
- `pickle` - For loading T5 embeddings  
- `pathlib.Path` - For better path handling

The modified script maintains compatibility with the existing training pipeline while supporting your local dataset structure.
