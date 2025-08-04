#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate",
#   "pydantic",
#   "qwen-vl-utils",
#   "rich",
#   "torch",
#   "torchcodec",
#   "torchvision",
#   "transformers>=4.51.3",
#   "vllm",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# ///

"""Run inference on a model with a given prompt.

Example:

```shell
./inference.py --prompt 'Please describe the video.' --videos video.mp4
```
"""

import argparse
import pathlib

import vllm
import pydantic
import qwen_vl_utils
import transformers
from rich import print
import yaml

ROOT = pathlib.Path(__file__).parents[1].resolve()

class Prompt(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    system_prompt: str = pydantic.Field(default="", description="System prompt")
    user_prompt: str = pydantic.Field(default="", description="User prompt")

class VisionConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    fps: int = pydantic.Field(default=1, description="FPS of the video")
    max_pixels: int = pydantic.Field(default=81920, description="Max pixels of the video")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, nargs="*", help="Image paths")
    parser.add_argument("--videos", type=str, nargs="*", help="Video paths")
    parser.add_argument("--prompt", type=str, default=f"{ROOT}/prompts/caption_reason.yaml", help="Path to prompt yaml file")
    parser.add_argument("--vision", type=str, help="Vision config json file")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason1-7B",
        help="Model name (https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7)",
    )
    args = parser.parse_args()

    images: list[str] = args.images or []
    videos: list[str] = args.videos or []
    if args.vision is not None:
        vision_config = VisionConfig.model_validate_json(open(args.vision, "rb").read())
    else:
        vision_config = VisionConfig()
    prompt_config = Prompt.model_validate(yaml.safe_load(open(args.prompt, "rb")))
    
    # Create messages
    user_content = []
    if prompt_config.user_prompt:
        user_content.append({"type": "text", "text": prompt_config.user_prompt})
    for image in images:
        user_content.append(
            {"type": "image", "image": image, "max_pixels": vision_config.max_pixels}
        )
    for video in videos:
        user_content.append(
            {
                "type": "video",
                "video": video,
                # HACK
                "fps": 4,
                # "fps": vision_config.fps,
                # "max_pixels": vision_config.max_pixels,
            }
        )
    messages = []
    if prompt_config.system_prompt:
        messages.append({"role": "system", "content": prompt_config.system_prompt})
    messages.append({"role": "user", "content": user_content})
    print("Messages:", messages)

    llm = vllm.LLM(
        model=args.model,
        limit_mm_per_prompt={"image": 10, "video": 10},
    )

    # Process messages
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained(args.model)
    )
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(messages, return_video_kwargs=True)

    # Run inference
    if False:
        generation_config = transformers.GenerationConfig(
            do_sample=True,
            max_new_tokens=4096,
            repetition_penalty=1.05,
            temperature=0.6,
            top_p=0.95,
        )

        model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype="auto", device_map="auto"
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, generation_config=generation_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_text = outputs[0]
    else:
        sampling_params = vllm.SamplingParams(
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
            max_tokens=4096,
        )
    
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        output_text = outputs[0].outputs[0].text

    print(f"\n\n{output_text}")


if __name__ == "__main__":
    main()
