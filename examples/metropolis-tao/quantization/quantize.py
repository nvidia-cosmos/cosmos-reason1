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

"""Quantization script for Cosmos-RL.

Example:

```shell
cosmos-rl-quantize --model_path nvidia/Cosmos-Reason1-7B --results_dir /results/quantized_model
```
"""

import argparse
import base64
import json
import logging
import os
import shutil
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import requests
import torch
from datasets import load_dataset, Dataset as HFDataset
from huggingface_hub import snapshot_download
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot
from llmcompressor.utils import dispatch_for_generation
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from cosmos_rl.utils.decorators import monitor_status
from cosmos_rl.utils.tao_status_logger import log_tao_status
from cosmos_rl.utils.lora_utils import merge_lora_model
from nvidia_tao_core.loggers.logging import get_status_logger, Status, Verbosity

# Constants
COMPONENT_NAME = "Cosmos-RL Quantization"
SEPARATOR = "-" * 50

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_and_tokenize(example, processor, max_sequence_length):
    """Apply chat template and tokenize inputs."""
    # preprocess image
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_image = f"data:image;base64,{encoded_image_text}"
    # Create conversation with image and text prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base64_image},
                {
                    "type": "text",
                    "text": "What does the image show? Please describe in detail.",
                },
            ],
        }
    ]
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Process vision info - extract images from messages
    image_inputs = []
    for message in messages:
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if item.get("type") == "image":
                    if item["image"].startswith("data:image"):
                        # Decode base64 image
                        base64_data = item["image"].split(",")[1]
                        image_data = base64.b64decode(base64_data)
                        image = Image.open(BytesIO(image_data))
                        image_inputs.append(image)
                    elif item["image"].startswith("http"):
                        # Download image from URL
                        response = requests.get(item["image"])
                        image = Image.open(BytesIO(response.content))
                        image_inputs.append(image)
    # tokenize
    # Note: Don't use truncation with multimodal inputs as it can cut image tokens
    return processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        padding=False,
        truncation=False,  # Disable truncation for multimodal inputs
    )


def data_collator(batch):
    """Oneshot data collator for multimodal inputs."""
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def load_custom_dataset(annotation_path: str, media_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load custom dataset from annotation JSON file.

    Args:
        annotation_path: Path to annotation JSON file
        media_dir: Optional directory containing media files (prepended to relative paths)

    Returns:
        List of dataset samples with 'image' field containing PIL Image objects
    """
    logger.info(f"Loading custom dataset from: {annotation_path}")

    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    logger.info(f"Loaded {len(annotations)} annotations")

    samples = []
    for item in annotations:
        # Extract image or video path(s)
        images = item.get("image") or item.get("images")
        videos = item.get("video") or item.get("videos")

        # For videos, extract first frame as image for calibration
        if videos and not images:
            try:
                import cv2

                # Handle single video or list of videos
                if isinstance(videos, str):
                    video_path = videos
                elif isinstance(videos, list) and len(videos) > 0:
                    video_path = videos[0]
                else:
                    logger.warning(f"Invalid video format for sample: {item.get('id', 'unknown')}")
                    continue

                # Prepend media_dir if provided
                if media_dir:
                    video_path = os.path.join(media_dir, video_path)

                # Extract first frame from video
                if os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    cap.release()

                    if ret:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)
                        samples.append({"image": image})
                    else:
                        logger.warning(f"Failed to read first frame from video: {video_path}")
                else:
                    logger.warning(f"Video file not found: {video_path}")

            except ImportError:
                logger.warning(f"cv2 not available - skipping video sample: {item.get('id', 'unknown')}")
                continue
            except Exception as e:
                logger.warning(f"Failed to extract frame from video: {e}")
                continue

        elif images:
            # Handle single image or list of images
            if isinstance(images, str):
                image_path = images
            elif isinstance(images, list) and len(images) > 0:
                image_path = images[0]  # Use first image for calibration
            else:
                logger.warning(f"Invalid image format for sample: {item.get('id', 'unknown')}")
                continue

            # Prepend media_dir if provided
            if media_dir:
                image_path = os.path.join(media_dir, image_path)

            # Load image
            try:
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                    samples.append({"image": image})
                else:
                    logger.warning(f"Image file not found: {image_path}")
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                continue
        else:
            logger.warning(f"Skipping sample without image or video: {item.get('id', 'unknown')}")
            continue

    logger.info(f"Successfully loaded {len(samples)} samples from custom dataset")
    return samples


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
    parser = argparse.ArgumentParser(description="Cosmos-RL Quantization Script")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or name of the model to quantize (e.g., nvidia/Cosmos-Reason1-7B)"
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="lmms-lab/flickr30k",
        help="HuggingFace dataset ID for calibration (e.g., lmms-lab/flickr30k). Leave empty to use custom dataset."
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test[:512]",
        help="Dataset split for calibration (e.g., test[:512]). Only used with HuggingFace datasets."
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=None,
        help="Path to custom annotation JSON file. Use this instead of dataset_id for local datasets."
    )
    parser.add_argument(
        "--media_dir",
        type=str,
        default=None,
        help="Directory containing media files for custom dataset. Paths in annotations are relative to this directory."
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=512,
        help="Number of calibration samples to use"
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/results",
        help="Directory to save the quantized model and TAO logs"
    )
    parser.add_argument(
        "--quantization_scheme",
        type=str,
        default="FP8_DYNAMIC",
        choices=["FP8_DYNAMIC", "W8A8", "W8A16", "W4A16"],
        help="Quantization scheme to use (FP8_DYNAMIC for W8A8)"
    )
    parser.add_argument(
        "--smoothing_strength",
        type=float,
        default=0.8,
        help="SmoothQuant smoothing strength (0.0 to 1.0)"
    )
    parser.add_argument(
        "--skip_test_generation",
        type=str_to_bool,
        default=False,
        help="Skip test generation after quantization"
    )
    parser.add_argument(
        "--enable_lora",
        type=str_to_bool,
        default=False,
        help="Enable LoRA model merging (required if model_path is a LoRA checkpoint)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Base model path for LoRA merging (required if enable_lora is True)"
    )

    return parser.parse_args()


def run_quantization(args):
    """
    Run the quantization pipeline.

    Args:
        args: Parsed command line arguments
    """

    # Get status logger for TAO integration
    s_logger = get_status_logger()

    try:
        s_logger.write(
            status_level=Status.RUNNING,
            message="Starting quantization pipeline...",
            verbosity_level=Verbosity.INFO
        )

        # Create save directory
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Quantized model will be saved to: {results_dir}")

        # Log quantization start to TAO
        log_tao_status(
            data={
                "quantization_status": "started",
                "model_path": args.model_path,
                "results_dir": str(results_dir),
                "quantization_scheme": args.quantization_scheme,
                "num_calibration_samples": args.num_calibration_samples,
                "lora_enabled": args.enable_lora,
                "base_model_path": args.base_model_path if args.enable_lora else None
            },
            component_name=COMPONENT_NAME
        )

        logger.info(SEPARATOR)
        logger.info("STARTING QUANTIZATION PIPELINE")
        logger.info(SEPARATOR)
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Quantization Scheme: {args.quantization_scheme}")

        # Log dataset information
        if args.annotation_path and os.path.exists(args.annotation_path):
            logger.info(f"Dataset: Custom (annotation: {args.annotation_path})")
            if args.media_dir:
                logger.info(f"Media Directory: {args.media_dir}")
        else:
            logger.info(f"Dataset: HuggingFace ({args.dataset_id}, split: {args.dataset_split})")

        logger.info(f"Calibration Samples: {args.num_calibration_samples}")
        logger.info(f"Save Directory: {results_dir}")
        logger.info(SEPARATOR)

        # Handle LoRA merging if enabled
        model_path = args.model_path
        if args.enable_lora:
            if not args.base_model_path:
                raise ValueError("--base_model_path is required when --enable_lora is specified")

            s_logger.write(
                status_level=Status.RUNNING,
                message=f"Merging LoRA adapter from {model_path} with base model {args.base_model_path}",
                verbosity_level=Verbosity.INFO
            )
            logger.info(f"Merging LoRA adapter from {model_path} with base model {args.base_model_path}")

            model_path = merge_lora_model(args.model_path, args.base_model_path)

            s_logger.write(
                status_level=Status.SUCCESS,
                message=f"LoRA merging complete. Using merged model: {model_path}",
                verbosity_level=Verbosity.INFO
            )
            logger.info(f"LoRA merging complete. Using merged model: {model_path}")
        else:
            # Check if the model is in PEFT/LoRA format
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                error_msg = (
                    f"The checkpoint at {model_path} is in PEFT/LoRA format. "
                    "Please enable LoRA merging by setting --enable_lora=true and providing --base_model_path."
                )
                s_logger.write(
                    status_level=Status.FAILURE,
                    message=error_msg,
                    verbosity_level=Verbosity.ERROR
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Load model
        s_logger.write(
            status_level=Status.RUNNING,
            message=f"Loading model: {model_path}",
            verbosity_level=Verbosity.INFO
        )
        logger.info(f"Loading model: {model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", trust_remote_code=True, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        s_logger.write(
            status_level=Status.SUCCESS,
            message="Model loaded successfully",
            verbosity_level=Verbosity.INFO
        )

        # Load calibration dataset (either HuggingFace or custom)
        use_custom_dataset = args.annotation_path and os.path.exists(args.annotation_path)

        if use_custom_dataset:
            # Load custom dataset from annotation file
            s_logger.write(
                status_level=Status.RUNNING,
                message=f"Loading custom calibration dataset: {args.annotation_path}",
                verbosity_level=Verbosity.INFO
            )
            logger.info(f"Loading custom calibration dataset: {args.annotation_path}")

            custom_samples = load_custom_dataset(args.annotation_path, args.media_dir)

            # Convert to HuggingFace Dataset format
            ds = HFDataset.from_list(custom_samples)

            # Shuffle and limit samples
            ds = ds.shuffle(seed=42)
            if len(ds) > args.num_calibration_samples:
                ds = ds.select(range(args.num_calibration_samples))

            logger.info(f"Using {len(ds)} samples from custom dataset")
        else:
            # Load HuggingFace dataset
            s_logger.write(
                status_level=Status.RUNNING,
                message=f"Loading HuggingFace calibration dataset: {args.dataset_id}",
                verbosity_level=Verbosity.INFO
            )
            logger.info(f"Loading HuggingFace calibration dataset: {args.dataset_id}")
            ds = load_dataset(args.dataset_id, split=args.dataset_split)
            ds = ds.shuffle(seed=42)

        s_logger.write(
            status_level=Status.RUNNING,
            message="Preprocessing calibration dataset...",
            verbosity_level=Verbosity.INFO
        )
        logger.info("Preprocessing dataset...")
        ds = ds.map(
            lambda x: preprocess_and_tokenize(x, processor, args.max_sequence_length),
            remove_columns=ds.column_names,
        )

        s_logger.write(
            status_level=Status.SUCCESS,
            message=f"Dataset preprocessed: {len(ds)} samples",
            verbosity_level=Verbosity.INFO
        )

        # Recipe for quantization
        recipe = [
            # SmoothQuant helps make activations easier to quantize
            SmoothQuantModifier(
                smoothing_strength=args.smoothing_strength,
                mappings=[
                    # Map attention and MLP layers
                    [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
                    [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
                ],
            ),
            # Apply quantization
            GPTQModifier(
                targets="Linear",
                scheme=args.quantization_scheme,
                ignore=[
                    "lm_head",
                    "re:visual.*",
                    "re:model.visual.*",
                ],  # Don't quantize vision encoder and lm_head
            ),
        ]

        s_logger.write(
            status_level=Status.RUNNING,
            message=f"Starting {args.quantization_scheme} quantization process...",
            verbosity_level=Verbosity.INFO
        )
        logger.info(f"Starting {args.quantization_scheme} quantization process...")
        logger.info("This may take a while depending on your GPU...")

        # Perform oneshot quantization
        oneshot(
            model=model,
            tokenizer=args.model_path,
            dataset=ds,
            recipe=recipe,
            max_seq_length=args.max_sequence_length,
            num_calibration_samples=args.num_calibration_samples,
            trust_remote_code_model=True,
            data_collator=data_collator,
            sequential_targets=[
                "Qwen2_5_VLDecoderLayer"
            ],  # Sequential processing for memory efficiency
        )

        s_logger.write(
            status_level=Status.SUCCESS,
            message="Quantization complete!",
            verbosity_level=Verbosity.INFO
        )
        logger.info("Quantization complete!")

        # Test the quantized model with a sample generation (unless skipped)
        if not args.skip_test_generation:
            logger.info("========== SAMPLE GENERATION ==============")
            s_logger.write(
                status_level=Status.RUNNING,
                message="Running test generation...",
                verbosity_level=Verbosity.INFO
            )

            dispatch_for_generation(model)
            # Test with a sample image
            test_url = "http://images.cocodataset.org/train2017/000000231895.jpg"
            test_image = Image.open(BytesIO(requests.get(test_url).content))
            test_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_url},
                        {"type": "text", "text": "Please describe the animal in this image\n"},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(test_messages, add_generation_prompt=True)
            inputs = processor(
                text=[prompt],
                images=[test_image],
                padding=False,
                max_length=args.max_sequence_length,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")

            logger.info("Generating response...")
            output = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
            generated_text = processor.decode(output[0], skip_special_tokens=True)

            logger.info(f"Generated: {generated_text}")
            logger.info("==========================================")

        # Save the quantized model
        s_logger.write(
            status_level=Status.RUNNING,
            message=f"Saving quantized model to: {results_dir}",
            verbosity_level=Verbosity.INFO
        )
        logger.info(f"\nSaving quantized model to: {results_dir}")

        model.save_pretrained(results_dir)

        # Copy processor files
        # use snapshot_download or copy files to make sure correct files are being stored with the checkpoint
        # processor files are incorrect after save_pretrained
        if not (model_path := Path(args.model_path)).exists():
            snapshot_download(
                repo_id=args.model_path,
                ignore_patterns=["config.json", "*.safetensors*"],
                local_dir=results_dir,
            )
        else:
            files_to_copy = [
                f
                for f in model_path.glob("*")
                if not (f.name == "config.json" or "safetensors" in f.name)
            ]
            for file in files_to_copy:
                shutil.copy(file, results_dir / file.name)

        s_logger.write(
            status_level=Status.SUCCESS,
            message=f"Quantized model saved successfully to: {results_dir}",
            verbosity_level=Verbosity.INFO
        )

        # Prepare KPI data
        kpi_data = {
            "quantization_status": "completed",
            "model_path": args.model_path,
            "results_dir": str(results_dir),
            "quantization_scheme": args.quantization_scheme,
            "num_calibration_samples": args.num_calibration_samples,
            "smoothing_strength": args.smoothing_strength,
            "lora_enabled": args.enable_lora,
            "base_model_path": args.base_model_path if args.enable_lora else None
        }

        # Log final results to TAO
        log_tao_status(
            data=kpi_data,
            component_name=COMPONENT_NAME
        )

        # Print summary
        print("\n" + "=" * 60)
        print("QUANTIZATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Model: {args.model_path}")
        print(f"Quantization Scheme: {args.quantization_scheme}")
        print(f"Saved to: {results_dir}")
        print()
        print(f"Note: {args.quantization_scheme} quantization provides:")
        if args.quantization_scheme == "FP8_DYNAMIC":
            print("- 8-bit weight quantization reduces model size")
            print("- 8-bit activation quantization speeds up inference")
        print()
        print("To use the quantized model with vLLM:")
        print("  from vllm import LLM")
        print(f'  model = LLM("{results_dir}")')
        print("  # vLLM will automatically handle quantized inference")
        print("=" * 60)

        s_logger.write(
            status_level=Status.SUCCESS,
            message=f"Quantization completed successfully. Model saved to: {results_dir}",
            verbosity_level=Verbosity.INFO
        )

    except KeyboardInterrupt:
        s_logger.write(
            status_level=Status.FAILURE,
            message="Quantization was interrupted by user (Ctrl+C)",
            verbosity_level=Verbosity.WARNING
        )
        log_tao_status(
            data={
                "quantization_status": "interrupted",
                "error": "User interrupted"
            },
            component_name=COMPONENT_NAME
        )
        raise

    except Exception as e:
        error_msg = f"Quantization failed: {str(e)}"
        s_logger.write(
            status_level=Status.FAILURE,
            message=error_msg,
            verbosity_level=Verbosity.ERROR
        )
        log_tao_status(
            data={
                "quantization_status": "failed",
                "error": str(e)
            },
            component_name=COMPONENT_NAME
        )
        logger.error(error_msg)
        raise


@monitor_status(name="Cosmos-RL Quantization", mode="quantize")
def main():
    """Main entry point for the cosmos-rl-quantize command."""
    args = parse_args()

    # Execute the quantization function with status monitoring
    run_quantization(args)


if __name__ == "__main__":
    main()

