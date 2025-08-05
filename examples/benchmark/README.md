# Cosmos Reason1 Benchmark Example

This guide provides instructions for evaluating models on the [Cosmos Reason1 Benchmark](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-Benchmark)

## Minimum Requirements

- 1 GPU with 24GB memory
- TODO: ?GB disk space.

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
  ```

Install the package:

```shell
cd cosmos-reason1/examples/benchmark
just install
source .venv/bin/activate
```

## Prepare Dataset

### Get Access

- [AgiBotWorld-Beta on Hugging Face](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/tree/main)

```shell
hf auth login
```

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
>   - Video clips for AgiBot-World, BridgeData V2, and HoloAssist must be downloaded manually in the next step (optional).

[Optional] To download the full dataset, run (this will take a very long time):

```bash
./tools/eval/process_raw_data.py \
  --data_dir data \
  --task benchmark
```

## Run Evaluation

Configure evaluation settings by editing [`configs/evaluate.yaml`](configs/evaluate.yaml).

Run evaluation:

```bash
python tools/eval/evaluate.py \
    --config configs/robovqa.yaml \
    --data_dir data \
    --results_dir results
```

### Benchmark Scoring

This step computes benchmark accuracy metrics from prediction results stored in a specified directory. It is used to evaluate model performance on datasets such as **RoboVQA**.

#### About the Evaluation

The evaluation uses **accuracy** as the primary metric, comparing model predictions against ground-truth answers. Accuracy is computed as:

> **Accuracy = (# correct predictions) / (total questions)**

For open-ended questions, a prediction is considered correct if it exactly matches the ground truth (case-insensitive string match). For multiple-choice questions, the selected option is compared against the correct choice.

> **Note:** These scoring rules follow common practices in VLM QA literature, but users are encouraged to adapt or extend them for specific use cases (e.g., partial credit, VQA-style soft accuracy).

#### Usage

Run the following command to compute accuracy:

```bash
./tools/eval/calculate_accuracy.py --result_dir results
```
