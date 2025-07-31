<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Paper](https://arxiv.org/abs/2503.15558) | [Website](https://research.nvidia.com/labs/dir/cosmos-reason1/) | [HuggingFace](https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7)

Cosmos-Reason1 is a suite of models, ontologies, and benchmarks that we develop with the goal of enabling multimodal LLMs to generate physically grounded responses. We release one multimodal LLMs: Cosmos-Reason1-7B post-trained with Physical AI SFT, and Physical AI reinforcement learning. We define ontologies for physical common sense and embodied reasoning, and also build benchmarks to evaluate Physical AI reasoning capabilities of multimodal LLMs.

## News
* 2025-06-11: We enhance the model’s capability on judging the physical plausibility of a video. See [this tutorial](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/video_critic.md) for details.
* 2025-05-17: We release model weights and training data under [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7).

## Model

* [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B)

## Inference

Install system dependencies:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Login to huggingface:

```shell
uv tool install -U "huggingface_hub[cli]"
hf auth login
```

Example scripts:

* [🤗 Transformers](scripts/inference.py)

  ```shell
  ./scripts/inference.py --prompt 'Please describe the video.' --videos assets/sample.mp4
  ```

* [vLLM](scripts/inference_vllm.py)

  ```shell
  ./scripts/inference_vllm.py
  ```

Example output:

<table>
  <tr>
    <td>
      <p>User prompt: Is it safe to turn right?</p>
      <img src="assets/sample.gif" alt="Demo GIF" width="400" /><br/>
    </td>
    <td style="vertical-align: top; padding-left: 20px;">
      <!-- Text on the right -->
      <p>Assistant response: </p>
      <p>
&lt;think&gt;
Okay, let's see. The user is asking if it's safe to turn right based on the video provided. First, I need to ...
The key points here are the parked vehicles. There are cars parked on both sides of the road, ...
&lt;/think&gt;

&lt;answer&gt;
Based on the video, turning right may not be entirely safe due to the following factors: ...
&lt;/answer&gt;
      </p>
    </td>
  </tr>
  <tr>
    <td>
      <p>User prompt: Analyze the video for any artifacts or anomalies.</p>
      <img src="examples/video_critic/assets/generation_3.gif" alt="Demo GIF" width="400" /><br/>
    </td>
    <td style="vertical-align: top; padding-left: 20px;">
      <!-- Text on the right -->
      <p>Assistant response:</p>
      <p>
&lt;component name="Orange Bottle Interaction"&gt;<br>
&lt;analysis&gt;
The left robot arm approaches the orange bottle and makes contact with it. However, instead of grasping the bottle firmly, the gripper wraps around it in an awkward manner, failing to secure a proper hold ... suggesting a glitch in the interaction physics. &lt;/analysis&gt;<br>
&lt;/component&gt;<br>
...
      </p>
    </td>
  </tr>
</table>

## Tutorials

* [Using Cosmos-Reason1 as Video Critic for Rejection Sampling](examples/video_critic/README.md)
* [Post-training Cosmos-Reason1 using Cosmos-Reason1-SFT-Dataset](examples/cosmos_reason1_sft_dataset/README.md)

## Post-Training

The [nvidia-cosmos/cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl)  repository is an async post-training framework specialized for Supervised Fine-Tuning (SFT) and Reinforcement Learning with Human Feedback (RLHF). It prioritizes performance, scalability, and fault tolerance.

For detailed instructions on running SFT and PPO training, please refer to our [User Guide](docs/UserGuide.md).

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
