# Cosmos Reason1 Benchmark

This guide provides instructions for evaluating models on the [Cosmos Reason1 Benchmark](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-Benchmark)

## Requirements

- 0.5TB disk space.

## Setup

Install the package:

```shell
cd cosmos-reason1/examples/cosmos_reason1_benchmark
just install
source .venv/bin/activate
```

## Prepare Dataset

### Get Access

- [AgiBotWorld-Beta on Hugging Face](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/tree/main)

```shell
hf auth login
```

### Download Dataset

Download annotations and sample video clips using the script below:

```bash
hf download --repo-type dataset nvidia/Cosmos-Reason1-Benchmark --local-dir data/benchmark
for file in data/tmp/**/*.tar.gz; do tar -xzf "$file" -C "$(dirname "$file")"; done
```

> **Note:**
> This script downloads:
>
> - ✅ Annotations for:
>   - `AV` # For autonomous vehicles' general description, driving difficulty, and notice
>   - [RoboVQA](https://robovqa.github.io/) # Videos, instructions, and question-answer pairs of agents (robots, humans, humans-with-grasping-tools) executing a task.
>   - [AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World) # A wide range of real-life tasks for robot manipulation
>   - [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) # A wide array of robotic manipulation behaviors
>   - [HoloAssist Dataset](https://holoassist.github.io/) # Crucial first-person perspectives that provide natural and immersive understanding of human actions
> - ✅ Video clips for:
>   - `AV`
>   - `RoboVQA`
>   - ⚠️ Video clips for AgiBot-World, BridgeData V2, and HoloAssist must be downloaded manually in the next step (optional).

Run the following script to download and preprocess video clips:

```bash
./tools/eval/process_raw_data.py \
  --data_dir data \
  --task benchmark
```

## Run Evaluation on Benchmarks

This step walks you through running evaluations on your model using the provided script.

### Configure Evaluation

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

### Run Evaluation

Run the evaluation on the **RoboVQA** dataset:

```bash
# Set tensor parallelism size (adjust as needed)
export TP_SIZE=4

# Run the evaluation script
python tools/eval/evaluate.py \
    --config tools/eval/configs/robovqa.yaml \
    --data_dir data \
    --results_dir results
```

*Tip:* You can also use `--model_name` to specify either a Hugging Face model name or a local safetensors folder path mentioned above.

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
./tools/eval/calculate_accuracy.py --result_dir results --dataset robovqa
```

- `--result_dir`: Path to the directory containing the model's prediction results. This should match the `--result_dir` used during evaluation in `evaluate.py`.
- `--dataset`: Name of the dataset to evaluate (e.g., `robovqa`, `av`, `agibot`, `bridgev2`, `holoassist`).
