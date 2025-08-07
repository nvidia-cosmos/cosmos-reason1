#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate",
#   "matplotlib",
#   "pillow",
#   "pydantic",
#   "pyyaml",
#   "qwen-vl-utils",
#   "rich",
#   "torch",
#   "torchcodec",
#   "torchvision",
#   "transformers>=4.51.3",
#   "vllm",
# ]
# [tool.uv]
# exclude-newer = "2025-08-05T00:00:00Z"
# [tool.uv.sources]
# qwen-vl-utils = {git = "https://github.com/spectralflight/Qwen2.5-VL.git", branch = "cosmos", subdirectory = "qwen-vl-utils"}
# ///

"""Run inference on a model with a given prompt.

Example:

```shell
./scripts/inference.py --prompt prompts/caption.yaml --videos assets/sample.mp4
```
"""

import functools
import os
import resource
import warnings

# Suppress warnings and core dumps
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import argparse
import collections
import pathlib
import textwrap

import matplotlib.font_manager as fm
import numpy as np
import pydantic
import qwen_vl_utils
import torch
import torchvision
import torchvision.transforms.functional
import transformers
import vllm
import yaml
from PIL import Image, ImageDraw, ImageFont
from rich import print
from rich.pretty import pprint

ROOT = pathlib.Path(__file__).parents[1].resolve()
SEPARATOR = "-" * 20


class Prompt(pydantic.BaseModel):
    """Config for prompt."""

    model_config = pydantic.ConfigDict(extra="forbid")

    system_prompt: str = pydantic.Field(default="", description="System prompt")
    user_prompt: str = pydantic.Field(default="", description="User prompt")


def pprint_dict(d: dict, name: str):
    """Pretty print a dictionary."""
    pprint(collections.namedtuple(name, d.keys())(**d), expand_all=True)


def tensor_to_pil_images(video_tensor: torch.Tensor) -> list[Image.Image]:
    """Convert a video tensor to a list of PIL images.

    Args:
        video_tensor: Tensor with shape (C, T, H, W) or (T, C, H, W)

    Returns:
        List of PIL images
    """
    # Check tensor shape and convert if needed
    if video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:  # (C, T, H, W)
        # Convert to (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)

    # Convert to numpy array with shape (T, H, W, C)
    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()

    # Ensure values are in the right range for PIL (0-255, uint8)
    if video_np.dtype == np.float32 or video_np.dtype == np.float64:
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

    # Convert each frame to a PIL image
    pil_images = [Image.fromarray(frame) for frame in video_np]

    return pil_images


def pil_images_to_tensor(images: list[Image.Image]) -> torch.Tensor:
    """Convert a list of PIL images to a video tensor.

    Args:
        pil_images: List of PIL images

    Returns:
        Tensor with shape (T, C, H, W)
    """
    return torch.stack(
        [torchvision.transforms.functional.pil_to_tensor(image) for image in images],
        dim=0,
    )


@functools.cache
def get_overlay_font_path() -> str:
    """Return the path to the font for overlaying text on images."""
    # Use DejaVu Sans Mono font for better readability
    return fm.findfont(fm.FontProperties(family="DejaVu Sans Mono", style="normal"))


def overlay_text(
    images: list[Image.Image],
    fps: float,
    border_height: int = 28,  # this is due to patch size of 28
    temporal_path_size: int = 2,  # Number of positions to cycle through
    font_size: int = 20,
    font_color: str = "white",
) -> tuple[list[Image.Image], list[float]]:
    """Overlay text on a list of PIL images with black border.

    The timestamp position cycles through available positions.

    Args:
        images: List of PIL images to process
        fps: Frames per second
        border_height: Height of the black border in pixels (default: 28)
        temporal_path_size: Number of positions to cycle through (default: 2)
        font_size: Font size for the text (default: 20)
        font_color: Color of the text (default: "white")

    Returns:
        List of PIL images with text overlay
        List of timestamps
    """

    font = ImageFont.truetype(get_overlay_font_path(), font_size)

    # Process each image
    processed_images = []

    for i, image in enumerate(images):
        # Get original dimensions
        width, height = image.size

        # Create new image with black border at the bottom
        new_height = height + border_height
        new_image = Image.new("RGB", (width, new_height), color="black")

        # Paste original image at the top
        new_image.paste(image, (0, 0))

        # Draw text on the black border
        draw = ImageDraw.Draw(new_image)

        # Calculate timestamp for current frame
        total_seconds = i / fps
        text = f"{total_seconds:.2f}s"

        # Get text dimensions
        try:
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(text, font=font)

        # Define available positions (cycling through horizontal positions)
        position_idx = i % temporal_path_size
        section_width = width // temporal_path_size

        # Calculate x position based on cycling position
        section_center_x = position_idx * section_width + section_width // 2
        text_x = section_center_x - text_width // 2

        # Ensure text doesn't go outside bounds
        text_x = max(0, min(text_x, width - text_width))

        # Center vertically in the border
        text_y = height + (border_height - text_height) // 2

        # Draw the single timestamp
        draw.text((text_x, text_y), text, fill=font_color, font=font)

        processed_images.append(new_image)

    return processed_images, [i / fps for i in range(len(images))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, nargs="*", help="Image paths")
    parser.add_argument("--videos", type=str, nargs="*", help="Video paths")
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Overlay timestamps on videos",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Path to prompt yaml file",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask the model (user prompt)",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable reasoning trace",
    )
    parser.add_argument(
        "--vision-config",
        type=str,
        default=f"{ROOT}/configs/vision_config.yaml",
        help="Path to vision config json file",
    )
    parser.add_argument(
        "--sampling-params",
        type=str,
        default=f"{ROOT}/configs/sampling_params.yaml",
        help="Path to sampling parameters yaml file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason1-7B",
        help="Model name or path (Cosmos-Reason1: https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        help="Model revision (branch name, tag name, or commit id)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    images: list[str] = args.images or []
    videos: list[str] = args.videos or []
    prompt_config = Prompt.model_validate(yaml.safe_load(open(args.prompt, "rb")))
    reasoning_config = Prompt.model_validate(
        yaml.safe_load(open(f"{ROOT}/prompts/reasoning.yaml", "rb"))
    )
    vision_kwargs = pydantic.TypeAdapter(qwen_vl_utils.VideoConfig).validate_python(
        yaml.safe_load(open(args.vision_config, "rb"))
    )
    sampling_kwargs = yaml.safe_load(open(args.sampling_params, "rb"))
    sampling_params = vllm.SamplingParams(**sampling_kwargs)
    if args.verbose:
        pprint_dict(vision_kwargs, "VisionConfig")
        pprint_dict(sampling_kwargs, "SamplingParams")

    # Create messages
    system_prompt = prompt_config.system_prompt
    if args.reasoning:
        system_prompt = f"{system_prompt}\n{reasoning_config.system_prompt}"
    if args.question:
        user_prompt = args.question
    else:
        user_prompt = prompt_config.user_prompt
    if not user_prompt:
        raise ValueError("No user prompt provided")
    user_content = []
    for image in images:
        user_content.append({"type": "image", "image": image} | vision_kwargs)
    for video in videos:
        user_content.append({"type": "video", "video": video} | vision_kwargs)
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": user_content})
    if args.verbose:
        pprint(conversation, expand_all=True)
    print(SEPARATOR)
    print("System:", textwrap.indent(system_prompt, "  "), sep="\n")
    print("User:", textwrap.indent(user_prompt, "  "), sep="\n")
    print(SEPARATOR)

    llm = vllm.LLM(
        model=args.model,
        revision=args.revision,
        limit_mm_per_prompt={"image": len(images), "video": len(videos)},
        enforce_eager=True,
    )

    # Process messages
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained(args.model)
    )
    prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        conversation, return_video_kwargs=True
    )
    if args.timestamp:
        video_inputs = [
            pil_images_to_tensor(overlay_text(tensor_to_pil_images(video), fps)[0])
            for video, fps in zip(video_inputs, video_kwargs["fps"])
        ]

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
    print(SEPARATOR)
    for output in outputs[0].outputs:
        output_text = output.text
        print("Assistant:", textwrap.indent(output_text, "  "), sep="\n")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
