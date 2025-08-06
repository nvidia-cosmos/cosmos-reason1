#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate",
#   "pydantic",
#   "qwen-vl-utils",
#   "msgspec",
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
./scripts/inference.py --prompt prompts/caption.yaml --videos assets/sample.mp4
```
"""

import os
import resource
import warnings

# Suppress warnings and core dumps
warnings.filterwarnings("ignore")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import argparse
import pathlib

import msgspec
import vllm
import pydantic
import qwen_vl_utils
import transformers
from rich import print
import yaml

ROOT = pathlib.Path(__file__).parents[1].resolve()


class Prompt(pydantic.BaseModel):
    """Config for prompt."""

    model_config = pydantic.ConfigDict(extra="forbid")

    system_prompt: str = pydantic.Field(default="", description="System prompt")
    user_prompt: str = pydantic.Field(default="", description="User prompt")


class VisionConfig(pydantic.BaseModel):
    """Config for vision processing.

    Source: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    nframes: int | None = pydantic.Field(
        None, description="Number of frames of the video"
    )

    fps: float | None = pydantic.Field(None, description="FPS of the video")
    min_frames: int | None = pydantic.Field(None, description="Min frames of the video")
    max_frames: int | None = pydantic.Field(None, description="Max frames of the video")

    min_pixels: int | None = pydantic.Field(
        None, description="Min pixels of the image/video"
    )
    max_pixels: int | None = pydantic.Field(
        None, description="Max pixels of the image/video"
    )
    total_pixels: int | None = pydantic.Field(
        None, description="Total pixels of the video"
    )

    resized_height: int | None = pydantic.Field(
        None, description="Resized height of the video"
    )
    resized_width: int | None = pydantic.Field(
        None, description="Resized width of the video"
    )

    video_start: float | None = pydantic.Field(
        None, description="Start time of the video"
    )
    video_end: float | None = pydantic.Field(None, description="End time of the video")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, nargs="*", help="Image paths")
    parser.add_argument("--videos", type=str, nargs="*", help="Video paths")
    parser.add_argument(
        "--prompt",
        type=str,
        default=f"{ROOT}/prompts/caption.yaml",
        help="Path to prompt yaml file",
    )
    parser.add_argument(
        "--vision-config",
        type=str,
        default=f"{ROOT}/configs/vision_config.json",
        help="Path to vision config json file",
    )
    parser.add_argument(
        "--generation-config",
        type=str,
        default=f"{ROOT}/configs/generation_config.json",
        help="Path to generation config json file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason1-7B",
        help="Model name (https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7)",
    )
    args = parser.parse_args()

    images: list[str] = args.images or []
    videos: list[str] = args.videos or []
    prompt_config = Prompt.model_validate(yaml.safe_load(open(args.prompt, "rb")))
    vision_config = VisionConfig.model_validate_json(
        open(args.vision_config, "rb").read()
    )
    vision_kwargs = vision_config.model_dump(exclude_none=True)
    sampling_params = msgspec.json.decode(
        open(args.generation_config, "rb").read(), type=vllm.SamplingParams
    )

    # Create messages
    user_content = []
    if prompt_config.user_prompt:
        user_content.append({"type": "text", "text": prompt_config.user_prompt})
    for image in images:
        user_content.append({"type": "image", "image": image} | vision_kwargs)
    for video in videos:
        user_content.append({"type": "video", "video": video} | vision_kwargs)
    messages = []
    if prompt_config.system_prompt:
        messages.append({"role": "system", "content": prompt_config.system_prompt})
    messages.append({"role": "user", "content": user_content})

    llm = vllm.LLM(
        model=args.model,
        limit_mm_per_prompt={"image": len(images), "video": len(videos)},
        enforce_eager=True,
    )

    # Process messages
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained(args.model)
    )
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        messages, return_video_kwargs=True
    )

    # Run inference
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
    print("-" * 20)
    for output in outputs[0].outputs:
        output_text = output.text
        print(f"{output_text}")
        print("-" * 20)


if __name__ == "__main__":
    main()
