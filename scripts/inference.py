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
#   "cosmos-reason1-utils",
#   "pyyaml",
#   "qwen-vl-utils",
#   "rich",
#   "torchcodec",
#   "transformers>=4.51.3",
#   "vllm",
# ]
# [tool.uv]
# exclude-newer = "2025-08-05T00:00:00Z"
# [tool.uv.sources]
# cosmos-reason1-utils = {path = "../cosmos_reason1_utils", editable = true}
# ///

"""Run inference on a model with a given prompt.

Example:

```shell
./scripts/inference.py --prompt prompts/caption.yaml --videos assets/sample.mp4
```
"""
# ruff: noqa: E402

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

import qwen_vl_utils
import transformers
import vllm
import yaml
from rich import print
from rich.pretty import pprint

from cosmos_reason1_utils.text import PromptConfig
from cosmos_reason1_utils.vision import (
    VisionConfig,
    overlay_text_on_tensor,
    save_tensor,
)

ROOT = pathlib.Path(__file__).parents[1].resolve()
SEPARATOR = "-" * 20


def pprint_dict(d: dict, name: str):
    """Pretty print a dictionary."""
    pprint(collections.namedtuple(name, d.keys())(**d), expand_all=True)


def create_conversation(
    *,
    system_prompt: str = "",
    user_prompt: str = "",
    images: list[str] | None = None,
    videos: list[str] | None = None,
    vision_config: VisionConfig,
) -> list[dict]:
    vision_kwargs = vision_config.model_dump(exclude_unset=True)

    user_content = []
    if images is not None:
        for image in images:
            user_content.append({"type": "image", "image": image} | vision_kwargs)
    if videos is not None:
        for video in videos:
            user_content.append({"type": "video", "video": video} | vision_kwargs)
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": user_content})
    return conversation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, nargs="*", help="Image paths")
    parser.add_argument("--videos", type=str, nargs="*", help="Video paths")
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Overlay timestamp on video frames",
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
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory for debugging",
    )
    args = parser.parse_args()

    images: list[str] = args.images or []
    videos: list[str] = args.videos or []

    # Load configs
    prompt_config = PromptConfig.model_validate(yaml.safe_load(open(args.prompt, "rb")))
    vision_kwargs = yaml.safe_load(open(args.vision_config, "rb"))
    vision_config = VisionConfig.model_validate(vision_kwargs)
    sampling_kwargs = yaml.safe_load(open(args.sampling_params, "rb"))
    sampling_params = vllm.SamplingParams(**sampling_kwargs)
    if args.verbose:
        pprint_dict(vision_kwargs, "VisionConfig")
        pprint_dict(sampling_kwargs, "SamplingParams")

    # Create conversation
    system_prompts = [open(f"{ROOT}/prompts/addons/english.txt", "r").read()]
    if prompt_config.system_prompt:
        system_prompts.append(prompt_config.system_prompt)
    if args.reasoning and "<think>" not in prompt_config.system_prompt:
        system_prompts.append(open(f"{ROOT}/prompts/addons/reasoning.txt", "r").read())
    system_prompt = "\n\n".join(system_prompts)
    if args.question:
        user_prompt = args.question
    else:
        user_prompt = prompt_config.user_prompt
    if not user_prompt:
        raise ValueError("No question provided.")
    conversation = create_conversation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        videos=videos,
        vision_config=vision_config,
    )
    if args.verbose:
        pprint(conversation, expand_all=True)
    print(SEPARATOR)
    print("System:")
    print(textwrap.indent(system_prompt.rstrip(), "  "))
    print("User:")
    print(textwrap.indent(user_prompt.rstrip(), "  "))
    print(SEPARATOR)

    # Create model
    llm = vllm.LLM(
        model=args.model,
        revision=args.revision,
        limit_mm_per_prompt={"image": len(images), "video": len(videos)},
        enforce_eager=True,
    )

    # Process inputs
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
        for i, video in enumerate(video_inputs):
            video_inputs[i] = overlay_text_on_tensor(video, fps=video_kwargs["fps"][i])
    if args.output:
        if image_inputs is not None:
            for i, image in enumerate(image_inputs):
                save_tensor(image, f"{args.output}/image_{i}")
        if video_inputs is not None:
            for i, video in enumerate(video_inputs):
                save_tensor(video, f"{args.output}/video_{i}")

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
        print("Assistant:")
        print(textwrap.indent(output_text.rstrip(), "  "))
    print(SEPARATOR)


if __name__ == "__main__":
    main()
