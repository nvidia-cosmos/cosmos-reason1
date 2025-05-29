# Cosmos-Reason1 Tools

This document provides usage instructions for scripts related to training and evaluating models on the Cosmos-Reason1 benchmark.



## Job recipes
Overall, cosmos_reason1 can support SFT/RL training with a broad range of models and parallelisms, users can configure these with a config file. For convenience, we have provided several pre-defined configs in the `configs` folder. e.g:

| Config File                                         | Policy TP | Policy FSDP | Policy PP | Rollout TP | Rollout PP | Num GPUs                     | Purpose |
| --------------------------------------------------- | --------- | ----------- | --------- | ---------- | ---------- | ---------------------------- | ------- |
| `cosmos-reason1-7b-tp2-sft.toml`                    | 2         | 1           | 1         | -          | -          | 2                            | SFT     |
| `cosmos-reason1-7b-p-fsdp1-tp2-r-tp2-pp1-grpo.toml` | 2         | 1           | 1         | 2          | 1          | 2 for policy,  2 for rollout | GRPO    |

SFT training requires using a **minimum of 2** GPUs, and RL training requires **at least 4** GPUs. Depending on the size of the model you are going to train, for **7B (or larger size)** models, GPUs with **80GB** of memory are required. For **3B (or smaller size)** models, GPUs with **>=32GB** memory are required.

You can customize your own training config based on the above recipes. For example, you can change the `epoch` and `train_batch_per_replica` to adjust the epochs number and batch size, and `save_freq` to adjust the checkpoint saving interval. To reduce the storage usage, you can reduce the number of `max_keep` in checkpoint config, which limits the number of saved checkpoint. Regularly delete intermediate checkpoints and tar archives not needed for recovery.

For the evaluation or inference, single GPU with at least 24GB memory is sufficient to run the `nvidia/Cosmos-Reason1-7B` model.

## Setup

### 🔧 Environment

You can either build docker image from source or install dependencies inside your python env:

#### 🐳 Option 1: Build docker image from source:
```bash
docker build -t cosmos-reason1-dev:dev .
docker run -it --gpus all --shm-size=24G -w /workspace/cosmos_reason1 cosmos-reason1-dev:dev
```

#### 🔨 Option 2: Run in your own environment
```bash
apt-get update && apt-get install redis-server
pip install -r requirements.txt
pip install -e .
```

### 📈 Monitor

We recommend that you to use wandb for training monitoring

1. Install wandb
```bash
pip install wandb
```

2. Login wandb, you can acquire your WANDB_API_KEY from [here](https://wandb.ai/authorize). Then you can login wandb by:

```bash
wandb login # Then enter your WANDB_API_KEY 
```

or you can add WANDB_API_KEY to your environment variables by adding the line to your shell config (e.g., `~/.bashrc`):
```bash
export WANDB_API_KEY=${WANDB_API_KEY}
```


3. Launch training with the following Training Scripts, you will see the wandb link in the logging:
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

In this example, we demonstrate how to launch SFT training for `nvidia/Cosmos-Reason1-7B` with TP=2 on 2 GPUs:

```shell
python tools/launch_all.py --config configs/cosmos-reason1/cosmos-reason1-7b-tp2-sft.toml
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

In this example, we demonstrate how to launch GRPO training for `nvidia/Cosmos-Reason1-7B` with `TP=2` & `FSDP=1`, and with rollout of `TP=2`, in total 4 GPUs:

```shell
python tools/launch_all.py --config configs/cosmos-reason1/cosmos-reason1-7b-p-fsdp1-tp2-r-tp2-pp1-grpo.toml
```
After training is done, the huggingface checkpoint gets saved to the directory `$output_dir`, which is similar to the SFT case. To evaluate the improved reasoning performance of this RL-trained model, please refer to the Evaluation section.

### Customized Reward Function

1. Register a new reward function with signature `(to_be_evaluated: str, reference: Union[str, None], *args, **kwargs) -> float`.
```python
from cosmos_reason1.dispatcher.algo.reward import register_reward_fn

def custom_reward_fn(to_be_evaluated: str, reference: Union[str, None], *args, **kwargs) -> float:
    ...

register_reward_fn("your_reward_fn_name", custom_reward_fn, override_toml=True)
```

if `override_toml` is set to `True`, the reward function will be registered and used no matter what is specified in the toml file.

2. (Optional) If `override_toml` is set to `False`, make sure to specify the reward function manually in the toml file:
```toml
[train.train_policy]
reward_fn = ["custom_reward_fn1", "custom_reward_fn2"...]
```

3. Specify customized controller launcher so that your registration get executed before controller launch.
  `export COSMOS_CONTROLLER_LAUNCHER=PATH_TO_LAUNCHER` 

Check [custom_reward_launcher.py](../tools/examples/custom_reward_launcher.py) for detail.


## 🚀 Inference
You may refer to the `inference.py` code snippet adopted from the [Qwen2.5-VL repo](https://github.com/QwenLM/Qwen2.5-VL/blob/main/README.md#inference-locally) to run inference with the Cosmos-Reason1 model.

```shell
python tools/eval/inference.py
```

## ✅ Evaluation Guide: `evaluate.py`

This document explains how to set up and run evaluation experiments using the `evaluate.py` script.



### 📦 Step 1: Download Benchmark Assets

Download annotations and sample video clips using the script below:

```bash
python tools/eval/download_hf_data.py \
    --target data \
    --task benchmark 
```

> **Note:**  
> This script downloads:  
> - ✅ Annotations for: 
>   - `AV` # For autonomous vehicles' general description, driving difficulty, and notice
>   - [RoboVQA](https://robovqa.github.io/) # Videos, instructions, and question-answer pairs of agents (robots, humans, humans-with-grasping-tools) executing a task.
>   - [AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World) # A wide range of real-life tasks for robot manipulation
>   - [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) # A wide array of robotic manipulation behaviors 
>   - [HoloAssist Dataset](https://holoassist.github.io/) # Crucial first-person perspectives that provide natural and immersive understanding of human actions
> - ✅ Video clips for:  
>   - `AV`
>   - `RoboVQA`
>   -  ⚠️ Video clips for AgiBot-World, BridgeData V2, and HoloAssist must be downloaded manually in the next step (optional).



### 🛠️ Step 2: (Optional) Download Remaining Video Clips

Follow the instructions in [RawDataDownload.md](RawDataDownload.md) to manually download and preprocess the AgiBot, BridgeV2, and HoloAssist datasets for evaluation.

### 🚀 Step 3: Run Evaluation on Benchmarks

This step walks you through running evaluations on your model using the provided script.


#### 📖 Configure Evaluation

You can configure evaluation settings by editing yaml file under `tools/eval/configs/`, take `robovqa.yaml` as example:

```yaml
datasets:
  - robovqa

model:
  model_name: nvidia/Cosmos-Reason1-7B # You can also replace the model_name by a safetensors folder path mentioned above
  tokenizer_model_name: qwen2.5-vl-7b
  dtype: bfloat16
  tp_size: 4
  max_length: 128000

evaluation:
  answer_type: reasoning
  num_processes: 80
  skip_saved: false
  fps: 4
  seed: 1

generation:
  max_retries: 10
  max_tokens: 1024
  temperature: 0.6
  repetition_penalty: 1.0
  presence_penalty: 0.0
  frequency_penalty: 0.0
```

#### 🔍 Run Evaluation

Run the evaluation on the **RoboVQA** dataset:

```bash
# Set tensor parallelism size (adjust as needed)
export TP_SIZE=4

# Run the evaluation script
PYTHONPATH=. python3 tools/eval/evaluate.py \
    --config tools/eval/configs/robovqa.yaml \
    --data_dir data \
    --results_dir results 
```

*Tip:* You can also use `--model_name` to specify either a Hugging Face model name or a local safetensors folder path mentioned above.


### 📊 Step 4: Benchmark Scoring

This step computes benchmark accuracy metrics from prediction results stored in a specified directory. It is used to evaluate model performance on datasets such as **RoboVQA**.

#### 📋 About the Evaluation

The evaluation uses **accuracy** as the primary metric, comparing model predictions against ground-truth answers. Accuracy is computed as:

> **Accuracy = (# correct predictions) / (total questions)**

For open-ended questions, a prediction is considered correct if it exactly matches the ground truth (case-insensitive string match). For multiple-choice questions, the selected option is compared against the correct choice.

> ⚠️ **Note:** These scoring rules follow common practices in VLM QA literature, but users are encouraged to adapt or extend them for specific use cases (e.g., partial credit, VQA-style soft accuracy).

#### 🔧 Usage

Run the following command to compute accuracy:

```bash
python tools/eval/calculate_accuracy.py --result_dir results --dataset robovqa
```

- `--result_dir`: Path to the directory containing the model's prediction results. This should match the `--result_dir` used during evaluation in `evaluate.py`.
- `--dataset`: Name of the dataset to evaluate (e.g., `robovqa`, `av`, `agibot`, `bridgev2`, `holoassist`).
