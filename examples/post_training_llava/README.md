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

Update the fields `annotation_path` and `media_path` in `configs/sft.toml`. `media_path` can be left as empty (`""`) if the paths in your annotation are absolute paths. 

Run SFT:

```shell
cosmos-rl --config configs/sft.toml scripts/custom_sft.py
```

The full config is saved to `outputs/sft/config.toml`.
