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

# Cosmos-Reason1 Post-Training Llava Example

This package provides a minimal Cosmos-Reason1 post-training example using the [Llava datasets](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md) format. You should first read the full post-training example, see [Cosmos-Reason1 Post-Training Full](../post_training/README.md).

## Setup

### Install

Prerequisites:

- [Setup](../post_training/README.md#setup)

Install the package:

```shell
cd examples/post_training_llava
just install
source .venv/bin/activate
```

## Example

Please update the fields `annotation_path` and `media_path` in `configs/sft.toml` to your custom dataset. `media_path` can be left as empty (`""`) if the paths in your annotation are absolute paths.

Here is one example of downloading the [Llava-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) dataset and [COCO](https://cocodataset.org/#home) images:

```shell
mkdir data && mkdir data/sft
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json -O data/sft/annotations.json
wget http://images.cocodataset.org/zips/train2017.zip -O data/sft/media.zip && unzip data/sft/media.zip -d data/sft/
```

Run SFT:

```shell
cosmos-rl --config configs/sft.toml scripts/custom_sft.py
```

The full config is saved to `outputs/sft/config.toml`.
