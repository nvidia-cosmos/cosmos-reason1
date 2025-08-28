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

import base64
import shutil
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path

import requests
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot
from llmcompressor.utils import dispatch_for_generation
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


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
    return processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        padding=False,
        max_length=max_sequence_length,
        truncation=True,
    )


def data_collator(batch):
    """Oneshot data collator for multimodal inputs."""
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="nvidia/Cosmos-Reason1-7B")
    parser.add_argument("--dataset_id", type=str, default="lmms-lab/flickr30k")
    parser.add_argument("--dataset_split", type=str, default="test[:512]")
    parser.add_argument("--num_calibration_samples", type=int, default=512)
    parser.add_argument("--max_sequence_length", type=int, default=2048)
    parser.add_argument("--save_dir", type=str, default="Cosmos-Reason1-7B-W8A8-FP8")
    args = parser.parse_args()

    # Oneshot arguments
    model_id = args.model_id
    dataset_id = args.dataset_id
    dataset_split = args.dataset_split
    num_calibration_samples = args.num_calibration_samples
    max_sequence_length = args.max_sequence_length
    save_dir = Path(args.save_dir)

    # Load model - nvidia/Cosmos-Reason1-7B (based on Qwen2.5-VL architecture)
    print(f"Loading model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", trust_remote_code=True, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Load dataset and preprocess.
    print(f"Loading calibration dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split=dataset_split)
    ds = ds.shuffle(seed=42)

    print("Preprocessing dataset...")
    ds = ds.map(
        lambda x: preprocess_and_tokenize(x, processor, max_sequence_length),
        remove_columns=ds.column_names,
    )

    # Recipe for W8A8 FP8 quantization
    # This uses both weight and activation quantization for maximum compression
    recipe = [
        # SmoothQuant helps make activations easier to quantize
        SmoothQuantModifier(
            smoothing_strength=0.8,
            mappings=[
                # Map attention and MLP layers
                [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
                [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
            ],
        ),
        # Apply W8A8 quantization
        GPTQModifier(
            targets="Linear",
            scheme="FP8_DYNAMIC",  # 8-bit weights AND 8-bit activations
            ignore=[
                "lm_head",
                "re:visual.*",
                "re:model.visual.*",
            ],  # Don't quantize vision encoder and lm_head
            group_size=128,  # Group size for GPTQ
        ),
    ]

    print("Starting W8A8 FP8 quantization process...")
    print("This provides maximum compression with 8-bit weights AND activations")
    print("This may take a while depending on your GPU...")
    # Perform oneshot quantization
    oneshot(
        model=model,
        tokenizer=model_id,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_sequence_length,
        num_calibration_samples=num_calibration_samples,
        trust_remote_code_model=True,
        data_collator=data_collator,
        sequential_targets=[
            "Qwen2_5_VLDecoderLayer"
        ],  # Sequential processing for memory efficiency
    )
    print("Quantization complete!")

    # Test the quantized model with a sample generation
    print("========== SAMPLE GENERATION ==============")
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
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    print("Generating response...")
    output = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    generated_text = processor.decode(output[0], skip_special_tokens=True)

    print(f"Generated: {generated_text}")
    print("==========================================")
    # Save the quantized model

    print(f"\nSaving quantized model to: {save_dir}")

    model.save_pretrained(save_dir)
    # use snapshot_download or copy files to make sure correct files are being stored with the checkpoint
    # processor files are incorrect after save_pretrained
    if not (model_path := Path(model_id)).exists():
        snapshot_download(
            repo_id=model_id,
            ignore_patterns=["config.json", "*.safetensors*"],
            local_dir=save_dir,
        )
    else:
        files_to_copy = [
            f
            for f in model_path.glob("*")
            if not (f.name == "config.json" and "safetensors" in f.name)
        ]
        for file in files_to_copy:
            shutil.copy(file, save_dir / file.name)

    print(f"\nQuantization complete! Model saved to: {save_dir}")
    print("\nNote: W8A8 provides maximum compression and speed:")
    print("- 8-bit weight quantization reduces model size")
    print("- 8-bit activation quantization speeds up inference")
    print("\nTo use the quantized model with vLLM:")
    print("  from vllm import LLM")
    print(f'  model = LLM("{save_dir}")')
    print("  # vLLM will automatically handle FP8 inference")
