<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Paper](https://arxiv.org/abs/2503.15558) | [Website](https://research.nvidia.com/labs/dir/cosmos-reason1/) | [HuggingFace](https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7) | [Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook)

NVIDIA Cosmos Reason – an open, customizable, 7B-parameter reasoning vision language model (VLM) for physical AI and robotics - enables robots and vision AI agents to reason like humans, using prior knowledge, physics understanding and common sense to understand and act in the real world. This model understands space, time, and fundamental physics, and can serve as a planning model to reason what steps an embodied agent might take next.
Cosmos Reason excels at navigating the long tail of diverse scenarios of the physical world with spatial-temporal understanding. Cosmos Reason is post-trained with physical common sense and embodied reasoning data with supervised fine-tuning and reinforcement learning. It uses chain-of-thought reasoning capabilities to understand world dynamics without human annotations.

## News

* 2025-10-28: We added [Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook), a collection of step-by-step recipes and post-training scripts to quickly build, customize, and deploy NVIDIA’s Cosmos world foundation models for robotics and autonomous systems.
* 2025-08-08: We added the [`cosmos-reason1-utils`](cosmos_reason1_utils/README.md) inference utilities package. Adds spatial-temporal reasoning inference. See [Inference](#inference) for example usage.
* 2025-08-1: We added support for spatial-temporal reasoning for city and industrial operations. See latest checkpoint [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B).
* 2025-06-11: We enhance the model’s capability on judging the physical plausibility of a video. See [this tutorial](examples/video_critic/README.md) for details.
* 2025-05-17: We release model weights and training data under [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7).

## Model

* [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B)

## Setup

> **This repository only contains documentation/examples/utilities. You do not need it to run inference. See [Inference example](scripts/inference_sample.py) for a minimal inference example. The following setup instructions are only needed to run the examples in this repository.**

Install system dependencies:

* [uv](https://docs.astral.sh/uv/getting-started/installation/)

  ```shell
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env
  ```

* [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

  ```shell
  uv tool install -U "huggingface_hub[cli]"
  hf auth login
  ```

Clone the repository:

```shell
git clone https://github.com/nvidia-cosmos/cosmos-reason1.git
cd cosmos-reason1
```

## Inference

Minimum Requirements:

* 1 GPU with 24GB memory

Cosmos-Reason1 is included in [`transformers>=4.51.3`](https://huggingface.co/docs/transformers/en/index).

We provide example inference scripts:

* [Minimal example](scripts/inference_sample.py)

  ```shell
  uv run scripts/inference_sample.py
  ```

* [Full example](scripts/inference.py)

  Caption the video:

  ```shell
  ./scripts/inference.py --prompt prompts/caption.yaml --videos assets/sample.mp4 -v
  ```

  Ask a question about the video with reasoning:

  ```shell
  ./scripts/inference.py --prompt prompts/question.yaml --question 'What are the potential safety hazards?' --reasoning --videos assets/sample.mp4 -v
  ```

  Temporally caption the video and save the input frames to `outputs/temporal_caption_text` for debugging:

  ```shell
  ./scripts/inference.py --prompt prompts/temporal_caption_text.yaml --videos assets/sample.mp4 --timestamp -v -o outputs/temporal_caption_text
  ```

  Configure inference by editing:

  * [Prompts](prompts/README.md)
  * [Sampling Parameters](configs/sampling_params.yaml)
  * [Vision Processor Config](configs/vision_config.yaml)

## Tutorials

* [Video Critic](examples/video_critic/README.md)
* Post-Training
  * [Full example](examples/post_training/README.md)
  * [Minimal Hugging Face example](examples/post_training_hf/README.md)
  * [Minimal Llava example](examples/post_training_llava/README.md)
* [Benchmark](examples/benchmark/README.md)

## Post-Training

The [nvidia-cosmos/cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) repository is an async post-training framework specialized for Supervised Fine-Tuning (SFT) and Reinforcement Learning with Human Feedback (RLHF). It prioritizes performance, scalability, and fault tolerance.

To support a custom dataset format, use the [minimal Hugging Face example](examples/post_training_hf/README.md) as a template.

## Additional Resources

The Cosmos-Reason1 model is based on the Qwen2.5-VL model architecture. Useful resources:

* [Repository](https://github.com/QwenLM/Qwen2.5-VL/blob/main/README.md)

## Post-Training quantization

To run PTQ `"vllm==0.9.2" "transformers>=4.53.1" "qwen-vl-utils[decord]" "llmcompressor>=0.6.0"` are required

[Full example](scripts/quantize_fp8.py)

```shell
./scripts/quantize_fp8.py --model_id 'nvidia/Cosmos-Reason1-7B' --save_dir 'Cosmos-Reason1-7B-W8A8-FP8'
```

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
