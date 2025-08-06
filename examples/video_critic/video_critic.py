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

"""Example script for using Cosmos Reason1 as a video critic.

Example:

```shell
./examples/video_critic/video_critic.py
```
"""


import argparse
import os
import base64
import xml.etree.ElementTree as ET

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT_CRITIC = """You are a helpful video analyzer. The goal is to identify artifacts and anomalies in the video.
Analyze the video carefully and answer the question according to the following template:

<think>
<overview>
[Brief description of the video.]
</overview>

<component name="Component 1 Name">
<analysis>
[Analysis or reasoning about this component.]
</analysis>
<anomaly>Yes | No</anomaly>
</component>

<component name="Component 2 Name">
<analysis>
[Analysis or reasoning about this component.]
</analysis>
<anomaly>Yes | No</anomaly>
</component>

<!-- Add more components as needed -->
</think>

<answer>
[Whether the video contains anomalies or artifacts. Answer "Yes" or "No".]
</answer>"""

USER_PROMPT_CRITIC = "Does the video contain any anomalies or artifacts?"

MODEL_PATH = "nvidia/Cosmos-Reason1-7B"


def parse_response(response):
    try:
        wrapped = f"<root>{response.strip()}</root>"
        root = ET.fromstring(wrapped)

        result = {"think": {}, "answer": None}

        # Parse <think> section
        think_element = root.find("think")
        if think_element is not None:
            # Parse overview
            overview = think_element.find("overview")
            result["think"]["overview"] = overview.text.strip() if overview is not None and overview.text else ""

            # Parse components
            result["think"]["components"] = []
            for comp in think_element.findall("component"):
                component_data = {"name": comp.get("name", "")}

                analysis = comp.find("analysis")
                component_data["analysis"] = analysis.text.strip() if analysis is not None and analysis.text else ""

                anomaly = comp.find("anomaly")
                component_data["anomaly"] = anomaly.text.strip() if anomaly is not None and anomaly.text else ""

                result["think"]["components"].append(component_data)

        # Parse <answer> section
        answer_element = root.find("answer")
        result["answer"] = answer_element.text.strip() if answer_element is not None and answer_element.text else ""

        return result
    except Exception:
        return None


def video_to_base64(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def build_html_report(video_path, responses):
    # Convert video to base64
    video_base64 = video_to_base64(video_path)
    mime_type = "video/mp4"

    # Parse responses
    parsed_responses = [parse_response(response) for response in responses]
    valid_responses = [r for r in parsed_responses if r is not None]

    # Count answers
    yes_count = sum(1 for r in valid_responses if r.get("answer", "").lower() == "yes")
    no_count = sum(1 for r in valid_responses if r.get("answer", "").lower() == "no")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cosmos-Reason1 Video Analysis Report - {os.path.basename(video_path)}</title>
    <style>
        body {{ font-family: sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        video {{ width: 100%; max-width: 600px; }}
        .red {{ background-color: #ffebee; color: #c62828; padding: 10px; margin: 5px 0; }}
        .green {{ background-color: #e8f5e8; color: #2e7d32; padding: 10px; margin: 5px 0; }}
        .trial {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ text-align: center; padding: 15px; }}
    </style>
</head>
<body>
    <h1>Cosmos-Reason1 Video Analysis Report</h1>
    <p>File: {os.path.basename(video_path)}</p>

    <h2>Video</h2>
    <video controls>
        <source src="data:{mime_type};base64,{video_base64}" type="{mime_type}">
    </video>

    <h2>Summary</h2>
    <div class="stats">
        <div class="stat red">
            <div style="font-size: 24px; font-weight: bold;">{yes_count}</div>
            <div>Anomaly Detected</div>
        </div>
        <div class="stat green">
            <div style="font-size: 24px; font-weight: bold;">{no_count}</div>
            <div>No Anomaly</div>
        </div>
        <div class="stat">
            <div style="font-size: 24px; font-weight: bold;">{len(valid_responses)}</div>
            <div>Total Responses</div>
        </div>
    </div>

    <h2>Detailed Analysis ({len(responses)} trials)</h2>
"""

    for i, (response, parsed) in enumerate(zip(responses, parsed_responses), 1):
        if parsed is None:
            html += f"""
    <div class="trial">
        <h3>Trial {i} - Failed to Parse</h3>
        <pre>{response}</pre>
    </div>
"""
        else:
            answer = parsed.get("answer", "").lower()
            answer_class = "red" if answer == "yes" else "green"

            html += f"""
    <div class="trial">
        <h3>Trial {i}</h3>
"""

            # Overview
            if parsed.get("think", {}).get("overview"):
                html += f"""
        <p><strong>Overview:</strong> {parsed["think"]["overview"]}</p>
"""

            # Components
            if parsed.get("think", {}).get("components"):
                for comp in parsed["think"]["components"]:
                    anomaly = comp.get("anomaly", "").lower()
                    comp_class = "red" if anomaly == "yes" else "green"
                    html += f"""
        <div class="{comp_class}">
            <strong>{comp.get('name', 'Unknown Component')} - {comp.get('anomaly', '')}</strong>
            <p>{comp.get('analysis', 'No analysis provided')}</p>
        </div>
"""

            # Final answer
            html += f"""
        <div class="{answer_class}">
            <strong>Final Answer: {parsed.get("answer", "No answer")}</strong>
        </div>
    </div>
"""

    html += """
</body>
</html>"""

    return html

def run_critic(args):
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 0, "video": 1},
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        n=args.num_trials,
        temperature=0.6,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.05,
        max_tokens=4096,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CRITIC},
        {"role": "user", "content": [
                {
                    "type": "video",
                    "video": args.video_path,
                    # Recommended settings for video critic:
                    "fps": 16,
                    "total_pixels": 8192 * 28 * 28,
                },
                {"type": "text", "text": USER_PROMPT_CRITIC},
            ]
        },
    ]
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

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
    generated_text = [output.text for output in outputs[0].outputs]

    return generated_text

def parse_args():
    parser = argparse.ArgumentParser(description="Run video critic inference and save html reports")
    parser.add_argument(
        "--video_path",
        type=str,
        default="assets/sample.mp4",
        help="Path to input video for critic",
    )
    # Rejection sampling settings
    parser.add_argument("--num_trials", type=int, default=4, help="Number of critic trials for each video")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generated_text = run_critic(args)
    html_content = build_html_report(args.video_path, generated_text)
    html_path = os.path.splitext(args.video_path)[0] + ".html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Generated HTML report: {html_path}")
