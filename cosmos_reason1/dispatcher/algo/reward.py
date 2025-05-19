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

import re
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from typing import Union, List
from cosmos_reason1.utils.constant import RewardFn
from cosmos_reason1.utils.logging import logger

math_comparer = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)


def format_reward_fn(to_be_evaluated: str, reference: Union[str, None]) -> float:
    try:
        pattern = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(pattern, to_be_evaluated, re.DOTALL)
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            return 0.0
        else:
            return 1.0
    except Exception as e:  # noqa: BLE001
        logger.debug("Exception in format_reward_func: %s", e)
        return 0.0


def boxed_math_reward_fn(to_be_evaluated: str, reference: Union[str, None]) -> float:
    """
    Compute the reward for the `to_be_evaluated` and `reference`.
    The reward is 1 if the `to_be_evaluated` is correct, otherwise 0.
    Answer are supposed to be in format: `\boxed{...}`.
    """
    try:
        score, _ = math_comparer([reference], [to_be_evaluated])
        return score
    except Exception:
        return 0.0


def single_choice_reward_fn(to_be_evaluated: str, reference: Union[str, None]) -> float:
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    reward = 0.0
    try:
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r"<answer>(.*?)</answer>", reference, re.DOTALL)
        ground_truth = sol_match.group(1).strip() if sol_match else reference.strip()

        # Extract answer from content if it has think/answer tags
        content_match = re.search(r"<answer>(.*?)</answer>", to_be_evaluated, re.DOTALL)
        student_answer = (
            content_match.group(1).strip() if content_match else to_be_evaluated.strip()
        )

        # Compare the extracted answers
        if student_answer.lower() == ground_truth.lower():
            reward = 1.0
    except Exception:
        reward = 0.0

    return reward


REWARD_FUNC_MAPPING = {
    RewardFn.BOXED_MATH: boxed_math_reward_fn,
    RewardFn.SINGLE_CHOICE: single_choice_reward_fn,
    RewardFn.FORMAT: format_reward_fn,
}


class Reward:
    def __init__(self, reward_funcs: List[str] = []):
        self.reward_funcs = []
        for name in reward_funcs:
            reward_func = RewardFn.from_string(name)
            if reward_func not in REWARD_FUNC_MAPPING:
                raise ValueError(f"Reward function {reward_func} not found in mapping.")
            self.reward_funcs.append(REWARD_FUNC_MAPPING[name])

    def compute_reward(self, to_be_evaluated: str, reference: Union[str, None]):
        total_reward = 0.0
        for func in self.reward_funcs:
            total_reward += func(to_be_evaluated, reference)
        return total_reward
