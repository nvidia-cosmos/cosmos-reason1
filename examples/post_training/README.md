# Cosmos Reason1 Post-Training Example

This guide provides instructions for post-training using [cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) on the Cosmos Reason1 [SFT](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-SFT-Dataset)/[RL](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-RL-Dataset) datasets.

- [cosmos-rl documentation](https://nvidia-cosmos.github.io/cosmos-rl/).

## Setup

### Install

Install system dependencies:

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

  ```shell
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env
  ```

- [just](https://github.com/casey/just?tab=readme-ov-file#installation)

  ```shell
  pkgm install just
  # or
  conda install -c conda-forge just
  ```

- [redis](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/)

  ```shell
  pkgm install redis-server
  # or
  conda install -c conda-forge redis-server
  ```

- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

  ```shell
  uv tool install -U "huggingface_hub[cli]"
  hf auth login
  ```

Install the package:

```shell
cd cosmos-reason1/examples/post_training
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

When you launch training, you will see the `wandb` link in the logging:

```bash
wandb: Currently logged in as: ${WANDB_USER_NAME} to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.11
wandb: Run data is saved locally in ./outputs/qwen2-5-3b-tp2-dpn-sft/20250515101157/wandb/run-20250515_101157-20250515101157
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run ./outputs/qwen2-5-3b-tp2-dpn-sft/20250515101157
wandb: ⭐️ View project at https://wandb.ai/${WANDB_USER_NAME}/${config.logging.project_name}
wandb: 🚀 View run at https://wandb.ai/${WANDB_USER_NAME}/${config.logging.project_name}/runs/20250515101157
```

Then you can view online visual training metrics, or check these data in the local wandb folder.

## Training Scripts

> **_NOTE:_**  Following the below training steps will trigger downloading around 200GB of model and dataset files from Hugging Face, please make sure your `~/.cache` directory (or set `HF_HOME` and `COSMOS_CACHE` environment variables to a directory that) has enough storage space.

## Supervised Fine-Tuning (SFT)

The SFT training can improve the model's capability on certain tasks with a similar distribution of the training dataset. E.g., training with `robovqa` dataset can improve the model's performance on the robotics-focused visual question answering scenarios.

### Minimum Requirements

- 4 GPUs with 80GB of memory
- 200GB of disk space

### Config

[Base config](configs/sft.toml)

Variants:


- 8 GPU

  ```toml
  [policy.parallelism]
  dp_shard_size = 8
  ```

### Run

In this example, we demonstrate how to launch SFT training for `nvidia/Cosmos-Reason1-7B` with `FSDP=2` on 2 GPUs:

```shell
cosmos-rl --config configs/sft.toml ./tools/dataset/cosmos_sft.py
```

After training finishes, the final output checkpoint can be found in the log:

```log
[rank0]:Exported safetensors to ./outputs/cosmos-reason1-7b-tp2-sft/20250516061336/safetensors/final
```

In this case, you will find the sft model checkpoint at `outputs/cosmos-reason1-7b-tp2-sft/20250516061336/safetensors/final`

## Reinforcement Learning (RL)

The RL training can improve the model's reasoning capability on certain tasks with the reasoning training dataset.

### Minimum Requirements

- 4 GPUs with 80GB of memory

### Config

[Base config](configs/rl.toml)

Config variants:

- 8 GPU

  ```toml
  [rollout.parallelism]
  tp_size = 4

  [policy.parallelism]
  dp_shard_size = 4
  ```

### Train

Run:

```shell
cosmos-rl --config configs/rl.toml tools/dataset/cosmos_grpo.py
```

After training is done, the huggingface checkpoint gets saved to the directory `$output_dir`, which is similar to the SFT case. To evaluate the improved reasoning performance of this RL-trained model, please refer to the Evaluation section.

## Evaluation

To evaluate the post-trained model, run the [Cosmos Reason1 Benchmark](../benchmark/README.md).
