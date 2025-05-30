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

import json
import os
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from cosmos_reason1.policy.config import GrpoConfig
from cosmos_reason1.policy.config import Config as CosmosConfig
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.util import basename_from_modelpath


def _disable_qwen_vl_utils_logger():
    import logging

    qwen_vl_utils_logger = logging.getLogger("qwen_vl_utils")
    # disable the qwen_vl_utils logger
    qwen_vl_utils_logger.setLevel(logging.WARNING)
    qwen_vl_utils_logger.propagate = False


_disable_qwen_vl_utils_logger()


def prepare_vl_dataset(
    grpo_config: GrpoConfig,
):
    """Prepare the dataset for the VLM model.

    Args:
        grpo_config: The GRPO configuration.

    Returns:
        path: The path to the dataset.
    """
    root_path = os.environ.get(
        "COSMOS_CACHE", os.path.join(os.path.expanduser("~"), ".cache/cosmos/")
    )
    video_clips_path = os.path.join(
        root_path,
        "datasets",
        basename_from_modelpath(grpo_config.dataset_name),
        grpo_config.dataset_subset,
        "video_clips",
    )

    if not os.path.exists(video_clips_path):
        raise FileNotFoundError(
            f"Dataset directory {video_clips_path} does not exist. Please check the dataset path."
        )

    mm_files_paths = {}
    for root, dirs, files in os.walk(video_clips_path):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov")):  # Common video extensions
                mm_files_paths[file] = os.path.join(root, file)

    return mm_files_paths


def construct_vl_input(
    prompts: list[str],
    config: CosmosConfig,
    processor: AutoProcessor,
    mm_files_paths: dict,
):
    """Construct the input for the model.

    Args:
        prompts: The json string prompts.
        config: The Cosmos configuration.
        processor: The processor for the VLM model.
        mm_files_list: The dict of video files's names to paths.

    Returns:
        A list of dictionaries containing the input for the model.
    """
    grpo_config = config.train.train_policy
    new_prompts = []

    for prompt in prompts:
        item = json.loads(prompt)
        if "video" in item or "image" in item:
            choices = item["qa_pairs"]["index2ans"]
            system_prompt = grpo_config.system_prompt

            user_prompt = (
                item["qa_pairs"]["question"]
                + "\n"
                + "\n".join([f"({i}) {choice}" for i, choice in choices.items()])
            )
            user_prompt += (
                "\nAnswer with the option's letter from the given choices directly."
            )
            user_prompt += "\nPlease answer the question in the following format: <think> your reasoning </think> <answer> your answer </answer>."

            if "video" in item:
                multi_modal_content = {
                    "type": "video",
                    "video": mm_files_paths[item["video"].split("/")[-1]],
                    "max_pixels": grpo_config.max_pixels,
                    "fps": grpo_config.fps,
                }
            else:
                multi_modal_content = {
                    "type": "image",
                    "image": mm_files_paths[item["image"].split("/")[-1]],
                }
            conversations = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        multi_modal_content,
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ]
            prompt = processor.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(conversations)

            if "video" in item:
                new_prompts.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"video": video_inputs},
                    }
                )
            else:
                new_prompts.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image_inputs},
                    }
                )
        else:
            logger.warning(
                f"[Rollout] The prompt {prompt} does not contain image or video data. "
                "Please check the input format."
            )
            continue

    return new_prompts
