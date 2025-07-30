# Nexar Collision Prediction

Post-training example using [Nexar Collision Prediction dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction).

```sh
# Download dataset
python examples/nexar_collision_prediction/download.py ~/datasets/nexar_collision_prediction/cosmos-rl

# Run SFT
cosmos-rl --config examples/nexar_collision_prediction/sft.toml ./tools/dataset/hf_cosmos_sft.py
```
