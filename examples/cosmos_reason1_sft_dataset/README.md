# Cosmos Reason1 SFT Dataset

This guide provides instructions for post-training on the (Cosmos Reason1 SFT Dataset)[https://huggingface.co/datasets/nvidia/Cosmos-Reason1-SFT-Dataset]

## Job recipes

Overall, cosmos_reason1 can support SFT/RL training with a broad range of models and parallelisms, users can configure these with a config file. For convenience, we have provided several pre-defined configs in the `configs` folder. e.g:

| Config File                                 | Policy TP | Policy FSDP | Policy PP | Rollout TP | Rollout PP | Num GPUs                     | Purpose |
| ------------------------------------------- | --------- | ----------- | --------- | ---------- | ---------- | ---------------------------- | ------- |
| `cosmos-reason1-7b-fsdp2-sft.toml`          | 1         | 2           | 1         | -          | -          | 2                            | SFT     |
| `cosmos-reason1-7b-p-fsdp2-r-tp2-grpo.toml` | 1         | 2           | 1         | 2          | 1          | 2 for policy,  2 for rollout | GRPO    |

SFT training requires using a **minimum of 2** GPUs, and RL training requires **at least 4** GPUs. Depending on the size of the model you are going to train, for **7B (or larger size)** models, GPUs with **80GB** of memory are required. For **3B (or smaller size)** models, GPUs with **>=32GB** memory are required.

You can customize your own training config based on the above recipes. For example, you can change the `epoch` and `train_batch_per_replica` to adjust the epochs number and batch size, and `save_freq` to adjust the checkpoint saving interval. To reduce the storage usage, you can reduce the number of `max_keep` in checkpoint config, which limits the number of saved checkpoint. Regularly delete intermediate checkpoints and tar archives not needed for recovery.

For the evaluation or inference, single GPU with at least 24GB memory is sufficient to run the `nvidia/Cosmos-Reason1-7B` model.

## Setup

### 🔧 Install

Install system dependencies:

* [uv](https://docs.astral.sh/uv/getting-started/installation/)

  ```shell
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env
  ```

* [just](https://github.com/casey/just?tab=readme-ov-file#installation)

  ```shell
  pkgm install just
  # or
  conda install -c conda-forge just
  ```

* [redis](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/)

  ```shell
  pkgm install redis-server
  # or
  conda install -c conda-forge redis-server
  ```

Install the package:

```shell
cd cosmos-reason1
just install
source .venv/bin/activate
```

### 📈 Monitor

We recommend that you to use wandb for training monitoring

1. Login wandb, you can acquire your WANDB_API_KEY from [here](https://wandb.ai/authorize). Then you can login wandb by:

```bash
wandb login # Then enter your WANDB_API_KEY
```

or you can add WANDB_API_KEY to your environment variables by adding the line to your shell config (e.g., `~/.bashrc`):
```bash
export WANDB_API_KEY=${WANDB_API_KEY}
```

2. Launch training with the following Training Scripts, you will see the wandb link in the logging:
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

### 🔩 Huggingface Access

To get access to Cosmos-SFT/RL datasets, you can add your HF_TOKEN to the environment variables by adding the line to your shell config (e.g., `~/.bashrc`):
```bash
export HF_TOKEN=${HF_TOKEN}
```

### 📝 Checkpoints Management
We support various types of checkpoints, e.g. basic checkpoints for training resume, huggingface safetensors for convient usage. We also support upload our checkpoints to huggingface and s3.

If you want to upload to huggingface, make sure your HF_TOKEN mentioned above have the **write access**.

If you want to upload to s3, you need to add the following variables to your environment:
```bash
export AWS_ENDPOINT_URL='your-endpoint-url' # Optional
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
export AWS_DEFAULT_REGION='your-s3-region'
```

Then you can configure the checkpoint settings in the toml file as follows:
```toml
[train.ckpt]
## Basic checkpoint
enable_checkpoint = true # Enable checkpointing for training. If set to False, no checkpoint will be saved.
save_freq = 100 # Checkpoint save frequency for training steps
max_keep = 2 # Maximum number of checkpoints to keep. If set to -1, all checkpoints will be kept.
save_mode = "async" # Checkpoint save mode for training steps, `async` is recommended

## Huggingface safetensors
export_safetensors = true # Whether to export a safetensors weight for huggingface usage, include related config files.
upload_hf = true # Whather to upload the final safetensors weight to huggingface.
hf_repo_name = "Cosmos-Reason1" # The huggingface repo name to upload the safetensors weight.

## S3
upload_s3 = true # Whether to upload the checkpoint and safetensors to S3. Default to False, set `final` will upload the final checkpoint, `all` will upload all checkpoints.
s3_bucket = 'your-s3-bucket' # The S3 bucket name to upload the checkpoint and safetensors weight.
s3_prefix = 'outputs' # The S3 prefix to upload the checkpoint and safetensors weight.
```

## 📘 Training Scripts

> **_NOTE:_**  Following the below training steps will trigger downloading around 200GB of model and dataset files from Hugging Face, please make sure your `~/.cache` directory (or set `HF_HOME` and `COSMOS_CACHE` environment variables to a directory that) has enough storage space.


### 🧠 Supervised Fine-Tuning (SFT)

The SFT training can improve the model's capability on certain tasks with a similar distribution of the training dataset. E.g., training with `robovqa` dataset can improve the model's performance on the robotics-focused visual question answering scenarios.


> **_NOTE:_**  We set the `nvidia/Cosmos-Reason1-7B` as the default base model of SFT, which is already SFT trained on the `nvidia/Cosmos-Reason1-SFT-Dataset`. We recommend you use your own dataset for SFT exploration.

In this example, we demonstrate how to launch SFT training for `nvidia/Cosmos-Reason1-7B` with `FSDP=2` on 2 GPUs:

```shell
cosmos-rl --config configs/cosmos-reason1-7b-fsdp2-sft.toml ./tools/dataset/cosmos_sft.py
```

After training finishes, the DCP checkpoint will be saved to `$output_dir`, and also with `huggingface` style model saved.

```
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

### 🔁 Reinforcement Learning (RL)

The RL training can improve the model's reasoning capability on certain tasks with the reasoning training dataset.

In this example, we demonstrate how to launch GRPO training for `nvidia/Cosmos-Reason1-7B` with `FSDP=2`, and with rollout of `TP=2`, in total 4 GPUs:

```shell
cosmos-rl --config configs/cosmos-reason1-7b-p-fsdp2-r-tp2-grpo.toml tools/dataset/cosmos_grpo.py
```
After training is done, the huggingface checkpoint gets saved to the directory `$output_dir`, which is similar to the SFT case. To evaluate the improved reasoning performance of this RL-trained model, please refer to the Evaluation section.
