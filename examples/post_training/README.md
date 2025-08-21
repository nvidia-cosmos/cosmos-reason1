# Cosmos-Reason1 Post-Training Example

This guide provides instructions for post-training Cosmos-Reason1 on the [SFT](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-SFT-Dataset)/[RL](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-RL-Dataset) datasets using [cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl).

- [cosmos-rl documentation](https://nvidia-cosmos.github.io/cosmos-rl/).

## Setup

### Installation

1. Perform the [Setup](../../README.md#setup) steps outlined in the main README.

2. Install system dependencies:

   - [redis](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/)

     ```shell
     pkgm install redis-server
     # or
     conda install -c conda-forge redis-server
     ```

3. Install the package:

```shell
cd examples/post_training
just install
source .venv/bin/activate
```

### Monitoring

We recommend using [wandb](https://wandb.ai/) to monitor training.

1. Acquire your [WANDB_API_KEY](https://wandb.ai/authorize).

2. Log in to wandb:

  ```bash
  uv tool install -U wandb
  wandb login
  ```

Now, when you run training, you will observe the `wandb` link in the logging:

```bash
wandb: 🚀 View run at https://wandb.ai/${WANDB_USER_NAME}/${config.logging.project_name}/runs/20250515101157
```

## Training

> **_NOTE:_** Following the below training steps will trigger downloading around 200GB of model and dataset files from Hugging Face. Ensure that your `~/.cache` directory has enough storage space or that the `HF_HOME` and `COSMOS_CACHE` environment variables are set to a directory with enough space.

### Supervised Fine-Tuning (SFT)

SFT training can improve model capability on tasks that have a similar distribution to that of the training dataset: For example, training with the `robovqa` dataset can improve performance with robotics-focused visual question answering scenarios.

#### Minimum Requirements

- 4 GPUs with 80GB of memory

#### Configuration

Configure settings by editing [configs/sft.toml](configs/sft.toml). Variants include the following:

- 8 GPU

  ```toml
  [policy.parallelism]
  dp_shard_size = 8
  ```

#### Training

Run training as follows:

```shell
cosmos-rl --config configs/sft.toml ./tools/dataset/cosmos_sft.py
```

After training finishes, the final output checkpoint can be found in the log:

```log
[rank0]:Exported safetensors to ./outputs/sft/20250516061336/safetensors/final
```

### Reinforcement Learning (RL)

RL training can improve model reasoning capability on certain tasks with the reasoning training dataset.

#### Minimum Requirements

- 4 GPUs with 80GB of memory

#### Configuration

Configure settings by editing [configs/rl.toml](configs/rl.toml). Variants include the following:

- 8 GPU

  ```toml
  [rollout.parallelism]
  tp_size = 4

  [policy.parallelism]
  dp_shard_size = 4
  ```

#### Training

Run training as follows:

```shell
cosmos-rl --config configs/rl.toml tools/dataset/cosmos_grpo.py
```

Similar to SFT training, the final output checkpoint can be found in the log.

## Evaluation

To evaluate the post-trained model, run the [Cosmos-Reason1 Benchmark](../benchmark/README.md).
