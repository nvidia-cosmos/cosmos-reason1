# Cosmos Reason1 Benchmark Example

This guide provides instructions for evaluating models on the [Cosmos Reason1 Benchmark](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-Benchmark)

## Minimum Requirements

- 1 GPU with 24GB memory

## Setup

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

- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

  ```shell
  uv tool install -U "huggingface_hub[cli]"
  hf auth login
  ```

Install the package:

```shell
cd cosmos-reason1/examples/benchmark
```

## Prepare Dataset

### Get Access

- [AgiBotWorld-Beta on Hugging Face](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/tree/main)

### Download Sample Dataset

Download annotations and sample video clips:

```bash
# Download
hf download --repo-type dataset nvidia/Cosmos-Reason1-Benchmark --local-dir data/benchmark
# Unpack
for file in data/tmp/**/*.tar.gz; do tar -xzf "$file" -C "$(dirname "$file")"; done
```

> **Note:**
> This downloads:
>
> - Annotations for:
>   - `AV` # For autonomous vehicles' general description, driving difficulty, and notice
>   - [RoboVQA](https://robovqa.github.io/) # Videos, instructions, and question-answer pairs of agents (robots, humans, humans-with-grasping-tools) executing a task.
>   - [AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World) # A wide range of real-life tasks for robot manipulation
>   - [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) # A wide array of robotic manipulation behaviors
>   - [HoloAssist Dataset](https://holoassist.github.io/) # Crucial first-person perspectives that provide natural and immersive understanding of human actions
> - Video clips for:
>   - `AV`
>   - `RoboVQA`

[Optional] Downloading the full dataset will take a very long time and requires multiple terabytes of disk space. To download, run:

```bash
./tools/eval/process_raw_data.py \
  --data_dir data \
  --task benchmark
```

> **Note:**
> This downloads:
>
> - Video clips for:
>   - AgiBot-World
>   - BridgeData V2
>   - HoloAssist

## Run Evaluation

Configure evaluation settings by editing [`configs/evaluate.yaml`](configs/evaluate.yaml).

Evaluate the model on the dataset:

```bash
./tools/eval/evaluate.py \
    --config configs/evaluate.yaml \
    --data_dir data \
    --results_dir results
```

### Compute Accuracy

Compute accuracy of the results:

```bash
./tools/eval/calculate_accuracy.py --result_dir results
```

The script compares model predictions against ground-truth answers. Accuracy is computed as:

> **Accuracy = (# correct predictions) / (total questions)**

For open-ended questions, a prediction is considered correct if it exactly matches the ground truth (case-insensitive string match). For multiple-choice questions, the selected option is compared against the correct choice.

> **Note:** These scoring rules follow common practices in VLM QA literature, but users are encouraged to adapt or extend them for specific use cases (e.g., partial credit, VQA-style soft accuracy).
