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
