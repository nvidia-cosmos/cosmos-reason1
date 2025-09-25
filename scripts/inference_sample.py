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

"""Minimal example of inference with Cosmos-Reason1.

Example:

```shell
uv run scripts/inference_sample.py
```
"""

from pathlib import Path

import qwen_vl_utils
import transformers

ROOT = Path(__file__).parents[2]
SEPARATOR = "-" * 20


def main():
    # Load model
    model_name = "nvidia/Cosmos-Reason1-7B"
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained(model_name)
    )

    # Create inputs
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"{ROOT}/assets/sample.mp4",
                    "fps": 4,
                    # 6422528 = 8192 * 28**2 = vision_tokens * (2*spatial_patch_size)^2
                    "total_pixels": 6422528,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

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

    # Run inference
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(SEPARATOR)
    print(output_text[0])
    print(SEPARATOR)


if __name__ == "__main__":
    main()
