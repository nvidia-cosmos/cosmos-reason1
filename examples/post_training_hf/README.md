> [!IMPORTANT]
> ## 🚀 [Cosmos 3 Has Arrived](https://github.com/NVIDIA/Cosmos)
>
> Cosmos 3 is NVIDIA's next-generation foundation model platform for Physical AI. Compared with Cosmos-Reason1, Cosmos 3 delivers substantially stronger physical reasoning capabilities while extending beyond reasoning to support world prediction, simulation, transfer, and action generation within a single unified model.
>
> Rather than relying on separate models for reasoning, prediction, transfer, and policy learning, a single Cosmos 3 model can understand the world, reason about physical interactions, predict future outcomes, transform observations across domains, and generate actions for embodied agents. This unified architecture enables stronger performance across a broad range of Physical AI applications, including robotics, autonomous vehicles, and smart spaces.
>
> This repository is no longer under active development and will receive only limited maintenance updates. Future model releases, features, documentation, and community support will be focused on Cosmos 3.
>
> 👉 Visit the new Cosmos home: https://github.com/NVIDIA/Cosmos
>
> There you will find the latest Cosmos 3 models, technical reports, tutorials, benchmarks, and ecosystem updates.
>
> Thank you for your support of Cosmos-Reason1. We encourage all users to migrate to Cosmos 3 for the latest state-of-the-art Physical AI capabilities.

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
