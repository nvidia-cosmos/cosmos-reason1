<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Paper](https://arxiv.org/abs/2503.15558) | [Website](https://research.nvidia.com/labs/dir/cosmos-reason1/) | [HuggingFace](https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7)

Cosmos-Reason1 is a suite of models, ontologies, and benchmarks that we develop with the goal of enabling multimodal LLMs to generate physically grounded responses. We release one multimodal LLMs: Cosmos-Reason1-7B post-trained with Physical AI SFT, and Physical AI reinforcement learning. We define ontologies for physical common sense and embodied reasoning, and also build benchmarks to evaluate Physical AI reasoning capabilities of multimodal LLMs.

## News

* 2025-08-1: We added support for spatial-temporal reasoning for city and industrial operations. See latest checkpoint [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B).
* 2025-06-11: We enhance the model’s capability on judging the physical plausibility of a video. See [this tutorial](examples/video_critic/README.md) for details.
* 2025-05-17: We release model weights and training data under [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7).

## Model

* [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B)

## Inference

Minimum Requirements:

* 1 GPU with 24GB memory

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

Example script:

```shell
./scripts/inference.py --prompt prompts/caption.yaml --videos assets/sample.mp4
```

## Prompts

We provide a set of task-specific prompts that are known to work well with Cosmos Reason1:

* Caption
  * [Caption (reasoning)](prompts/caption_reason.yaml)
  * [Caption (non-reasoning)](prompts/captions.yaml)
* Temporal
  * [Temporal Caption (json)](prompts/temporal_caption_json.yaml)
  * [Temporal Caption (text)](prompts/temporal_caption_text.yaml)
  * [Temporal Localization](prompts/temporal_localization.yaml)
* [Multiple Choice Questions](prompts/multiple_choice_questions.yaml)
* Analyzer
  * [Video Critic](prompts/video_critic.yaml)
  * [Video Analyzer](prompts/video_analyzer.yaml)
  * [Driving Analyzer](prompts/driving_analyzer.yaml)
* [Action Planning](prompts/action_planning.yaml)

## Tutorials

* [Video Critic for Rejection Sampling](examples/video_critic/README.md)
* [Post-Training](examples/post_training/README.md)
* [Benchmark](examples/benchmark/README.md)

## Post-Training

The [nvidia-cosmos/cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) repository is an async post-training framework specialized for Supervised Fine-Tuning (SFT) and Reinforcement Learning with Human Feedback (RLHF). It prioritizes performance, scalability, and fault tolerance.

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
