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
#   "qwen-vl-utils>=0.0.11",
#   "torchcodec>=0.6.0",
#   "torch>=2.7.1",
#   "transformers>=4.51.3",
#   "vllm>=0.10.1.1",
# ]
# [tool.uv.sources]
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

"""Minimal example of inference with Cosmos-Reason1.

Example:

```shell
./scripts/inference_sample.py --model_path /path/to/model --video test_video.mp4 --prompt "Describe this video."
```
"""

import argparse
from pathlib import Path
import shutil
import qwen_vl_utils
import transformers

from cosmos_rl.utils.decorators import monitor_status
from cosmos_rl.utils.lora_utils import merge_lora_model
from nvidia_tao_core.loggers.logging import get_status_logger, Status, Verbosity

ROOT = Path(__file__).parents[1]
SEPARATOR = "-" * 20


def str_to_bool(value):
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {value}')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cosmos-Reason1 inference script")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help="PyTorch data type (auto, float16, float32, bfloat16)"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device mapping strategy (auto, cpu, cuda, etc.)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for tensor parallelism (requires device_map='auto')"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="video",
        choices=["video", "image"],
        help="Input type (video or image)"
    )
    parser.add_argument(
        "--media",
        type=str,
        required=True,
        help="Path to video or image file"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second for video processing"
    )
    parser.add_argument(
        "--total_pixels",
        type=int,
        default=6422528,
        help="Total pixels for video processing (8192 * 28**2)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for the model"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Results directory"
    )
    parser.add_argument(
        "--enable_lora",
        type=str_to_bool,
        default=False,
        help="Enable LoRA model merging"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        help="Base model path for LoRA merging (required if enable_lora is True)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Apply monitor_status decorator with results_dir from args
    @monitor_status(name='Cosmos-RL Inference', mode='inference', results_dir=args.results_dir)
    def run_inference():
        s_logger = get_status_logger()

        s_logger.write(
            status_level=Status.RUNNING,
            message=f"Loading model from: {args.model_path}"
        )

        # Handle LoRA merging if enabled
        model_path = args.model_path
        adapter_backup_path = None
        if args.enable_lora:
            if not args.base_model_path:
                raise ValueError("--base_model_path is required when --enable_lora is specified")
            model_path = merge_lora_model(args.model_path, args.base_model_path)
            s_logger.write(
                status_level=Status.SUCCESS,
                message=f"LoRA merging enabled. Using merged model: {model_path}"
            )
        else:
            # If LoRA is disabled but adapter files exist, temporarily hide them during loading
            adapter_config_path = Path(model_path) / "adapter_config.json"
            if adapter_config_path.exists():
                adapter_backup_path = Path(model_path) / "adapter_config.json.bak"
                shutil.move(str(adapter_config_path), str(adapter_backup_path))
                s_logger.write(
                    status_level=Status.SUCCESS,
                    message="Temporarily hiding adapter files to load base model only"
                )

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
        }

        # Add max_memory constraint if num_gpus is specified
        if args.num_gpus is not None and args.num_gpus > 0:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Cannot use --num_gpus.")

            num_available_gpus = torch.cuda.device_count()
            if args.num_gpus > num_available_gpus:
                s_logger.write(
                    status_level=Status.FAILURE,
                    message=f"Requested {args.num_gpus} GPUs but only {num_available_gpus} available"
                )
                raise ValueError(f"Requested {args.num_gpus} GPUs but only {num_available_gpus} available")

            # Create max_memory dict to restrict to specified GPUs
            max_memory = {i: "0GB" for i in range(num_available_gpus)}
            for i in range(args.num_gpus):
                max_memory[i] = torch.cuda.get_device_properties(i).total_memory
            max_memory["cpu"] = "50GB"  # Allow CPU offloading if needed
            model_kwargs["max_memory"] = max_memory

            s_logger.write(
                status_level=Status.SUCCESS,
                message=f"Configuring model to use {args.num_gpus} GPU(s)"
            )

        # Load model
        try:
            model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, **model_kwargs
            )
            processor: transformers.Qwen2_5_VLProcessor = (
                transformers.AutoProcessor.from_pretrained(model_path)
            )

            s_logger.write(
                status_level=Status.SUCCESS,
                message="Model and processor loaded successfully"
            )
        finally:
            # Restore adapter config if it was temporarily hidden
            if not args.enable_lora and adapter_backup_path and adapter_backup_path.exists():
                adapter_config_path = Path(model_path) / "adapter_config.json"
                shutil.move(str(adapter_backup_path), str(adapter_config_path))
                s_logger.write(
                    status_level=Status.SUCCESS,
                    message="Restored adapter configuration files"
                )

        try:
            s_logger.write(
                status_level=Status.RUNNING,
                message=f"Preparing {args.type} input: {args.media}"
            )

            # Create inputs
            content = [
                {
                    "type": args.type,
                    "video" if args.type == "video" else "image": args.media,
                },
                {"type": "text", "text": args.prompt},
            ]

            # Add video-specific parameters if type is video
            if args.type == "video":
                content[0].update({
                    "fps": args.fps,
                    "total_pixels": args.total_pixels,
                })

            conversation = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            s_logger.write(
                status_level=Status.RUNNING,
                message="Processing inputs through chat template and vision pipeline"
            )

            # Process inputs
            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            input_ids_shape = inputs.input_ids.shape if hasattr(inputs, 'input_ids') else None
            s_logger.write(
                status_level=Status.SUCCESS,
                message=f"Input processing complete - Input shape: {input_ids_shape}"
            )

            s_logger.write(
                status_level=Status.RUNNING,
                message=f"Starting inference generation with max_new_tokens={args.max_new_tokens}"
            )

            # Run inference
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

            # Status callback: Inference complete
            generated_shape = generated_ids.shape if hasattr(generated_ids, 'shape') else None
            tokens_generated = generated_shape[1] - input_ids_shape[1] if (generated_shape and input_ids_shape and len(generated_shape) > 1 and len(input_ids_shape) > 1) else None
            s_logger.write(
                status_level=Status.RUNNING,
                message=f"Inference generation complete - Generated {tokens_generated} tokens"
            )

            s_logger.write(
                status_level=Status.RUNNING,
                message="Processing and decoding generated output"
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            output_length = len(output_text[0])
            s_logger.write(
                status_level=Status.SUCCESS,
                message=f"Results: {output_text[0][:100]}..." if output_length > 100 else f"Results: {output_text[0]}"
            )

            print(SEPARATOR)
            print(output_text[0])
            print(SEPARATOR)

        except KeyboardInterrupt as e:
            s_logger.write(
                status_level=Status.FAILURE,
                message="Inference was interrupted by user (Ctrl+C)",
                verbosity_level=Verbosity.WARNING
            )
            raise

        except Exception as e:
            s_logger.write(
                status_level=Status.FAILURE,
                message=f"Inference failed: {str(e)}",
                verbosity_level=Verbosity.ERROR
            )
            raise

    # Execute the decorated inference function
    run_inference()


if __name__ == "__main__":
    main()
