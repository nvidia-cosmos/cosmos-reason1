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
from transformers import PreTrainedTokenizer
from cosmos_reason1.policy.config import Config
from typing import Union, Callable
from cosmos_reason1.utils.constant import RewardFn
from cosmos_reason1.utils.logging import logger
from typing import Optional, List

math_comparer = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def format_reward_fn(
    to_be_evaluated: str, reference: Union[str, None], **kwargs
) -> float:
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


def overlong_reward_fn(
    to_be_evaluated: str,
    reference: Union[str, None],
    config: Config,
    tokenizer: PreTrainedTokenizer,
) -> float:
    """
    Reward function that checks if the completion is too long (DAPO).
    If the completion is longer than threshold, adopt a soft punishment.
    """
    overlong_buffer_len = config.train.train_policy.overlong_reward.buffer_length
    expected_len = config.rollout.max_response_length - overlong_buffer_len
    valid_response_length = len(
        tokenizer.encode(to_be_evaluated, add_special_tokens=False)
    )
    exceed_len = valid_response_length - expected_len
    overlong_penalty_factor = config.train.train_policy.overlong_reward.penalty_factor
    overlong_reward = min(
        -exceed_len / overlong_buffer_len * overlong_penalty_factor, 0
    )
    return overlong_reward


def direct_math_reward_fn(
    to_be_evaluated: str,
    reference: Union[str, None],
    **kwargs,
) -> float:
    """
    Compute the reward for the `to_be_evaluated` and `reference`.
    The reward is 1 if the `to_be_evaluated` is correct, otherwise -1.
    Answer pattern can be customized.
    """
    answer_pattern = kwargs.get("answer_pattern", r"(?i)Answer\s*:\s*([^\n]+)")
    # Extract answer from solution
    match = re.findall(answer_pattern, to_be_evaluated)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    if pred == reference:
        return 1.0
    else:
        return -1.0


def boxed_math_reward_fn(
    to_be_evaluated: str, reference: Union[str, None], **kwargs
) -> float:
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


def single_choice_reward_fn(
    to_be_evaluated: str, reference: Union[str, None], **kwargs
) -> float:
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


def gsm8k_reward_fn(
    to_be_evaluated: str, reference: Union[str, None], **kwargs
) -> float:
    def extract_solution(solution_str, method="strict"):
        assert method in ["strict", "flexible"]

        if method == "strict":
            # this also tests the formatting of the model
            solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
            if solution is None:
                final_answer = None
            else:
                final_answer = solution.group(0)
                final_answer = (
                    final_answer.split("#### ")[1].replace(",", "").replace("$", "")
                )
        elif method == "flexible":
            answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
            final_answer = None
            if len(answer) == 0:
                # no reward is there is no answer
                pass
            else:
                invalid_str = ["", "."]
                # find the last number that is not '.'
                for final_answer in reversed(answer):
                    if final_answer not in invalid_str:
                        break
        return final_answer

    try:
        return extract_solution(to_be_evaluated) == extract_solution(reference)
    except Exception:
        return 0.0


REWARD_FUNC_MAPPING = {
    RewardFn.DIRECT_MATH: direct_math_reward_fn,
    RewardFn.BOXED_MATH: boxed_math_reward_fn,
    RewardFn.SINGLE_CHOICE: single_choice_reward_fn,
    RewardFn.GSM8K: gsm8k_reward_fn,
    RewardFn.FORMAT: format_reward_fn,
    RewardFn.OVERLONG: overlong_reward_fn,
}


class Reward:
    def __init__(
        self,
        config: Config,
        tokenier: PreTrainedTokenizer,
        explicit_reward_fn: Optional[List[Callable]] = None,
    ):
        self.config = config
        self.tokenizer = tokenier
        if explicit_reward_fn:
            self.reward_funcs = (
                explicit_reward_fn
                if isinstance(explicit_reward_fn, list)
                else [explicit_reward_fn]
            )
            logger.info(
                f"[Reward] Using provided reward functions: {self.reward_funcs}, `config.train.train_policy.reward_function` will be ignored"
            )
        else:
            self.reward_funcs = []
            for name in config.train.train_policy.reward_function:
                reward_func = RewardFn.from_string(name)
            if reward_func not in REWARD_FUNC_MAPPING:
                raise ValueError(f"Reward function {reward_func} not found in mapping.")
            self.reward_funcs.append(REWARD_FUNC_MAPPING[name])
            logger.info(
                f"[Reward] Using reward functions: {config.train.train_policy.reward_function}"
            )

    def compute_reward(self, to_be_evaluated: str, reference: Union[str, None]):
        total_reward = 0.0
        for func in self.reward_funcs:
            total_reward += func(
                to_be_evaluated, reference, config=self.config, tokenizer=self.tokenizer
            )
        return total_reward
