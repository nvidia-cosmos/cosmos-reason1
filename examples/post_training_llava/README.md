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
