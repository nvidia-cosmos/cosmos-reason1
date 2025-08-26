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

# https://docs.astral.sh/uv/guides/scripts/#using-a-shebang-to-create-an-executable-file
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=1.10.1",
#   "cosmos-reason1-utils",
#   "pyyaml>=6.0.2",
#   "qwen-vl-utils>=0.0.11",
#   "rich",
#   "torch>=2.7.1",
#   "torchcodec>=0.6.0",
#   "transformers>=4.51.3",
#   "vllm>=0.10.1.1",
# ]
# [tool.uv.sources]
# cosmos-reason1-utils = {path = "../cosmos_reason1_utils", editable = true}
# torch = [
#   { index = "pytorch-cu128"},
# ]
# torchvision = [
#   { index = "pytorch-cu128"},
# ]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///

"""Full example of inference with Cosmos-Reason1.

Example:

```shell
./scripts/inference.py --prompt prompts/caption.yaml --videos assets/sample.mp4 -v
```
"""
# ruff: noqa: E402

from cosmos_reason1_utils.script import init_script

init_script()

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

from cosmos_reason1_utils.text import (
    PromptConfig,
    create_conversation,
    extract_tagged_text,
)
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
    prompt_kwargs = yaml.safe_load(open(args.prompt, "rb"))
    prompt_config = PromptConfig.model_validate(prompt_kwargs)
    vision_kwargs = yaml.safe_load(open(args.vision_config, "rb"))
    _vision_config = VisionConfig.model_validate(vision_kwargs)
    sampling_kwargs = yaml.safe_load(open(args.sampling_params, "rb"))
    sampling_params = vllm.SamplingParams(**sampling_kwargs)
    if args.verbose:
        pprint_dict(vision_kwargs, "VisionConfig")
        pprint_dict(sampling_kwargs, "SamplingParams")

    # Create conversation
    system_prompts = [open(f"{ROOT}/prompts/addons/english.txt").read()]
    if prompt_config.system_prompt:
        system_prompts.append(prompt_config.system_prompt)
    if args.reasoning and "<think>" not in prompt_config.system_prompt:
        if extract_tagged_text(prompt_config.system_prompt)[0]:
            raise ValueError(
                "Prompt already contains output format. Cannot add reasoning."
            )
        system_prompts.append(open(f"{ROOT}/prompts/addons/reasoning.txt").read())
    system_prompt = "\n\n".join(map(str.rstrip, system_prompts))
    if args.question:
        user_prompt = args.question
    else:
        user_prompt = prompt_config.user_prompt
    if not user_prompt:
        raise ValueError("No user prompt provided.")
    user_prompt = user_prompt.rstrip()
    conversation = create_conversation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        videos=videos,
        vision_kwargs=vision_kwargs,
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

    result, _ = extract_tagged_text(output_text)
    if args.verbose and result:
        pprint_dict(result, "Result")


if __name__ == "__main__":
    main()
