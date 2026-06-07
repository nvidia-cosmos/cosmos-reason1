> [!IMPORTANT]
> ## 🚀 [Cosmos 3 Has Arrived](https://github.com/NVIDIA/Cosmos)
>
> Cosmos 3 is NVIDIA's next-generation foundation model platform for Physical AI. Compared with Cosmos-Reason1, Cosmos 3 delivers substantially stronger physical reasoning capabilities while extending beyond reasoning to support world prediction, simulation, transfer, and action generation within a single unified model.
>
> Rather than relying on separate models for reasoning, prediction, transfer, and policy learning, a single Cosmos 3 model can understand the world, reason about physical interactions, predict future outcomes, transform observations across domains, and generate actions for embodied agents. This unified architecture enables stronger performance across a broad range of Physical AI applications, including robotics, autonomous vehicles, and smart spaces.
>
> This repository is no longer under active development and will receive only limited maintenance updates. Future model releases, features, documentation, and community support will be focused on Cosmos 3.
>
> 👉 Visit the new Cosmos home: https://github.com/NVIDIA/Cosmos
>
> There you will find the latest Cosmos 3 models, technical reports, tutorials, benchmarks, and ecosystem updates.
>
> Thank you for your support of Cosmos-Reason1. We encourage all users to migrate to Cosmos 3 for the latest state-of-the-art Physical AI capabilities.

# Prompts

We provide a set of task-specific prompt templates that are known to work well with Cosmos-Reason1:

* [Caption](caption.yaml)
* Question
  * [Question](question.yaml)
  * [Multiple Choice Question](multiple_choice_question.yaml)
* Temporal
  * [Temporal Caption (json)](temporal_caption_json.yaml)
  * [Temporal Caption (text)](temporal_caption_text.yaml)
  * [Temporal Localization](temporal_localization.yaml)
* Critic
  * [Video Analyzer](video_analyzer.yaml)
  * [Video Critic](video_critic.yaml)
* Domain Specific
  * [Action Planning](action_planning.yaml)
  * [AV](av.yaml)
  * [Driving](driving.yaml)
  * [Robot](robot.yaml)
* Utility
  * [Prompt Upsampler](prompt_upsampler.yaml)

## Addons

These are added to the system prompt to provide additional instructions:

* [Reasoning](addons/reasoning.txt)
* [English](addons/english.txt)

## Questions

Example questions:

* What are the potential safety hazards?
* Describe what is happening in this video/image.
* What objects do you see?
* What actions are being performed?
* Are there any people in this media?
* What is the robot doing?
* Is this demonstration successful?
* What could be improved in this process?
