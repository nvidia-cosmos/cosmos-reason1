# Cosmos-Reason1 Post-Training Using Hugging Face Dataset

This package provides a minimal Cosmos-Reason1 post-training example. For a full post-training example, see [Cosmos-Reason1 Post-Training](../post_training/).

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
./scripts/download_nexar_collision_prediction.py data/sft
```

Run SFT:

```shell
cosmos-rl --config configs/sft.toml ./scripts/sft.py
```

**TODO**: Add RL
