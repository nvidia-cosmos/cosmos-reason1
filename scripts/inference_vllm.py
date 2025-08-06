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

"""Example inference using vllm.

Example:

```shell
./inference_vllm.py
```
"""

from rich import print
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# You can also replace the MODEL_PATH by a safetensors folder path mentioned above
MODEL_PATH = "nvidia/Cosmos-Reason1-7B"

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.05,
    max_tokens=4096,
)

video_messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>.",
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": ("Is it safe to turn right?")},
            {
                "type": "video",
                "video": "assets/sample.mp4",
                "fps": 4,
            },
        ],
    },
]

# Here we use video messages as a demonstration
messages = video_messages

processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages, return_video_kwargs=True
)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs
if video_inputs is not None:
    mm_data["video"] = video_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
    # FPS will be returned in video_kwargs
    "mm_processor_kwargs": video_kwargs,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)
