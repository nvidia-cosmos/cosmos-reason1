# Preprocess Raw Datasets

This document guides you through manually downloading and preprocessing video clips from the AgiBot, BridgeV2, and HoloAssist datasets for evaluation.

## 📥 Get Access for AgiBot

- **Dataset**: [AgiBotWorld-Beta on Hugging Face](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/tree/main)  


## ⚙️ Run Preprocessing Script

Run the following script to download and preprocess video clips, take `holoassist` as example:

```bash
# Export HF_TOKEN to get access to Cosmos Reason dataset
export HF_TOKEN=...

python tools/eval/process_raw_data.py \
  --dataset holoassist \
  --data_dir data \
  --task benchmark
```

> 💡 Replace `holoassist` with `agibot` or `bridgev2` as needed.