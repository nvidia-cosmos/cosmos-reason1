# Cosmos-Reason1 Benchmark Example

This guide provides instructions for evaluating models on the [Cosmos-Reason1 Benchmark](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-Benchmark)

## Minimum Requirements

- 1 GPU with 24GB memory

## Setup

Prerequisites:

- [Setup](../../README.md#setup)

Change directory:

```shell
cd examples/benchmark
```

## Prepare Dataset

Request access:

- [AgiBotWorld-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/tree/main)

Download annotations and sample video clips:

```bash
# Download
hf download --repo-type dataset nvidia/Cosmos-Reason1-Benchmark --local-dir data/benchmark
# Unpack
for file in data/benchmark/**/*.tar.gz; do tar -xzf "$file" -C "$(dirname "$file")"; done
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

[Optional] Downloading the full dataset will take a very long time and requires multiple terabytes of disk space:

```bash
./tools/eval/process_raw_data.py --data_dir data --task benchmark
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
./tools/eval/evaluate.py --config configs/evaluate.yaml --data_dir data --results_dir outputs/benchmark
```

### Calculate Accuracy

Calculate accuracy of the results:

```bash
./tools/eval/calculate_accuracy.py --result_dir outputs/benchmark
```

The script compares model predictions against ground-truth answers:

> **Accuracy = (# correct predictions) / (total questions)**

For open-ended questions, a prediction is considered correct if it exactly matches the ground truth (case-insensitive string match). For multiple-choice questions, the selected option is compared against the correct choice.

> **Note:** These scoring rules follow common practices in VLM QA literature, but users are encouraged to adapt or extend them for specific use cases (e.g., partial credit, VQA-style soft accuracy).
