#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate",
#   "qwen-vl-utils",
#   "rich",
#   "torch",
#   "torchcodec",
#   "torchvision",
#   "transformers>=4.51.3",
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

import qwen_vl_utils
import transformers
from rich import print


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="User prompt message")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant. Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>.",
        help="System prompt message",
    )
    parser.add_argument("--images", type=str, nargs="*", help="Image paths")
    parser.add_argument("--videos", type=str, nargs="*", help="Video paths")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason1-7B",
        help="Model name (https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7)",
    )
    parser.add_argument(
        "--fps", type=int, default=1, help="Downsample video frame rate"
    )
    parser.add_argument(
        "--max-pixels", type=int, default=81920, help="Downsample media max pixels"
    )
    args = parser.parse_args()

    user_content = []
    for image in args.images or []:
        user_content.append(
            {"type": "image", "image": image, "max_pixels": args.max_pixels}
        )
    for video in args.videos or []:
        user_content.append(
            {
                "type": "video",
                "video": video,
                "fps": args.fps,
                "max_pixels": args.max_pixels,
            }
        )
    user_content.append({"type": "text", "text": args.prompt})
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": user_content})
    print("Messages:", messages)

    # Load the model
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained(args.model, use_fast=True)
    )

    generation_config = transformers.GenerationConfig(
        do_sample=True,
        max_new_tokens=4096,
        repetition_penalty=1.05,
        temperature=0.6,
        top_p=0.95,
    )

    # Process the messages
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Run inference
    generated_ids = model.generate(**inputs, generation_config=generation_config)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(f"\n\n{output_text[0]}")


if __name__ == "__main__":
    main()
