# Cosmos-Reason1 Post-Training Hugging Face Example

This package provides a minimal Cosmos-Reason1 post-training example using the [Hugging Face datasets](https://huggingface.co/docs/datasets/en/index) format. You should first read the full post-training example, see [Cosmos-Reason1 Post-Training Full](../post_training/README.md).

## Setup

### Install

Prerequisites:

- [Setup](../post_training/README.md#setup)

Install the package:

```shell
cd examples/post_training_hf
just install
source .venv/bin/activate
```

## Example

Download the [Nexar collision prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) dataset:

```shell
./scripts/download_nexar_collision_prediction.py data/sft --split "train[:10]"
```

Run SFT:

```shell
cosmos-rl --config configs/sft.toml scripts/custom_sft.py
```

The full config is saved to `outputs/sft/config.toml`.
