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

# Cosmos-Reason1 Post-Training Example

This guide provides instructions for post-training Cosmos-Reason1 on the [SFT](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-SFT-Dataset)/[RL](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-RL-Dataset) datasets using [cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl).

- [cosmos-rl documentation](https://nvidia-cosmos.github.io/cosmos-rl/).

## Setup

### Install

Prerequisites:

- [Setup](../../README.md#setup)

Install system dependencies:

- [redis](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/)

  ```shell
  conda install -c conda-forge redis-server
  ```

Install the package:

```shell
cd examples/post_training
just install
source .venv/bin/activate
```

### Monitor

[Optional] We recommend that you to use [wandb](https://wandb.ai/) for training monitoring.

1. Acquire your [WANDB_API_KEY](https://wandb.ai/authorize).
1. Login:

  ```bash
  uv tool install -U wandb
  wandb login
  ```

When you run training, you will see the `wandb` link in the logging:

```bash
wandb: 🚀 View run at https://wandb.ai/${WANDB_USER_NAME}/${config.logging.project_name}/runs/20250515101157
```

## Training

> **_NOTE:_** Following the below training steps will trigger downloading around 200GB of model and dataset files from Hugging Face, please make sure your `~/.cache` directory (or set `HF_HOME` and `COSMOS_CACHE` environment variables to a directory that) has enough storage space.

### Supervised Fine-Tuning (SFT)

The SFT training can improve the model's capability on certain tasks with a similar distribution of the training dataset. E.g., training with `robovqa` dataset can improve the model's performance on the robotics-focused visual question answering scenarios.

Minimum Requirements:

- 4 GPUs with 80GB of memory

Configure settings by editing [configs/sft.toml](configs/sft.toml). Variants:

- 8 GPU

  ```toml
  [policy.parallelism]
  dp_shard_size = 8
  ```

Run training:

```shell
cosmos-rl --config configs/sft.toml ./tools/dataset/cosmos_sft.py
```

After training finishes, the final output checkpoint can be found in the log:

```log
[rank0]:Exported safetensors to ./outputs/sft/20250516061336/safetensors/final
```

### Reinforcement Learning (RL)

The RL training can improve the model's reasoning capability on certain tasks with the reasoning training dataset.

Minimum Requirements:

- 4 GPUs with 80GB of memory

Configure settings by editing [configs/rl.toml](configs/rl.toml). Variants:

- 8 GPU

  ```toml
  [rollout.parallelism]
  tp_size = 4

  [policy.parallelism]
  dp_shard_size = 4
  ```

Run training:

```shell
cosmos-rl --config configs/rl.toml tools/dataset/cosmos_grpo.py
```

Similar to SFT training, the final output checkpoint can be found in the log.

## Evaluation

To evaluate the post-trained model, run the [Cosmos-Reason1 Benchmark](../benchmark/README.md).
