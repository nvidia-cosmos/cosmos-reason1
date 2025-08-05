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

Base config: [configs/sft.toml]

Variants:

- 8 GPU

  ```toml
  [policy.parallelism]
  dp_shard_size = 6
  ```

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

After training finishes, the DCP checkpoint will be saved to `$output_dir`, and also with `huggingface` style model saved.

```log
[rank1]:[cosmos] 2025-05-16 06:28:46,019 - cosmos - INFO - [Policy] Step: 95/95, [Policy] Loss: 0.87890625
[rank1]:[cosmos] 2025-05-16 06:28:46,020 - cosmos - INFO - [Policy] Training finished at step 95/95, saving final checkpoint in huggingface safetensors...
[rank0]:[cosmos] 2025-05-16 06:28:45,998 - cosmos - INFO - [Policy] Step: 95/95, [Policy] Loss: 0.87890625
[rank0]:[cosmos] 2025-05-16 06:28:45,999 - cosmos - INFO - [Policy] Training finished at step 95/95, saving final checkpoint in huggingface safetensors...
[rank0]:[cosmos] 2025-05-16 06:28:45,999 - cosmos - INFO - Prepare to exporting safetensors to ./outputs/cosmos-reason1-7b-tp2-sft/20250516061336/safetensors/final at rank 0
[rank0]:[cosmos] 2025-05-16 06:28:55,622 - cosmos - INFO - Saved chunk 0 to 00000.safetensors
[rank0]:[cosmos] 2025-05-16 06:29:03,829 - cosmos - INFO - Saved chunk 1 to 00001.safetensors
[rank0]:[cosmos] 2025-05-16 06:29:11,891 - cosmos - INFO - Saved chunk 2 to 00002.safetensors
[rank0]:[cosmos] 2025-05-16 06:29:21,191 - cosmos - INFO - Saved chunk 3 to 00003.safetensors
[rank0]:[cosmos] 2025-05-16 06:29:22,083 - cosmos - INFO -
[rank0]:
[rank0]:Exported safetensors to ./outputs/cosmos-reason1-7b-tp2-sft/20250516061336/safetensors/final
```

In this case, you will find the sft model checkpoint at `outputs/cosmos-reason1-7b-tp2-sft/20250516061336/safetensors/final`

```shell
root@node:~/ws# ls ./outputs/cosmos-reason1-7b-tp2-sft/20250516061336/safetensors/final -la
total 16211328
drwxr-xr-x 2 root root       4096 May 16 06:29 .
drwxr-xr-x 5 root root       4096 May 16 06:28 ..
-rw-r--r-- 1 root root 4171800072 May 16 06:28 00000.safetensors
-rw-r--r-- 1 root root 4195052544 May 16 06:29 00001.safetensors
-rw-r--r-- 1 root root 4195052632 May 16 06:29 00002.safetensors
-rw-r--r-- 1 root root 4022509168 May 16 06:29 00003.safetensors
-rw-r--r-- 1 root root        605 May 16 06:29 added_tokens.json
-rw-r--r-- 1 root root       1049 May 16 06:29 chat_template.json
-rw-r--r-- 1 root root       1459 May 16 06:29 config.json
-rw-r--r-- 1 root root    1671853 May 16 06:29 merges.txt
-rw-r--r-- 1 root root      49611 May 16 06:29 model.safetensors.index.json
-rw-r--r-- 1 root root        575 May 16 06:29 preprocessor_config.json
-rw-r--r-- 1 root root        613 May 16 06:29 special_tokens_map.json
-rw-r--r-- 1 root root   11421896 May 16 06:29 tokenizer.json
-rw-r--r-- 1 root root       5776 May 16 06:29 tokenizer_config.json
-rw-r--r-- 1 root root    2776833 May 16 06:29 vocab.json
```

To evaluate the improved performance of this sft model, please refer to the Evaluation section.

## Reinforcement Learning (RL)

The RL training can improve the model's reasoning capability on certain tasks with the reasoning training dataset.

### Minimum Requirements

- 4 GPUs with 80GB of memory

### Config

Base config: [configs/rl.toml]

Config variants:

- 6 GPU

  ```toml
  [train.train_policy]
  mini_batch = 4

  [rollout.parallelism]
  tp_size = 2

  [policy.parallelism]
  dp_shard_size = 4
  ```

### Train

Run:

```shell
cosmos-rl --config configs/rl.toml tools/dataset/cosmos_grpo.py
```

After training is done, the huggingface checkpoint gets saved to the directory `$output_dir`, which is similar to the SFT case. To evaluate the improved reasoning performance of this RL-trained model, please refer to the Evaluation section.
