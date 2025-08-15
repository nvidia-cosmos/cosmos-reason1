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

import concurrent.futures
import json
import logging as log
import os
import re

import attrs


#  ------------- output structure -------------
@attrs.define
class OutputStructure:
    """
    Represents the structure for storing input, model output, and evaluation results
    for a single item. Designed to be easily serialized to JSON.
    """

    # Input fields derived from the dataset
    datasource: str
    video_id: str
    output_json_fname: str  # Specifies the output file path for this item

    # Input fields related to the prompt/task
    prompt: str = ""
    correct_answer: str = ""  # The ground truth correct answer for evaluation

    # Fields for model output
    reasoning: str = ""  # The reasoning provided by the model
    answer: str = ""  # The final answer extracted from the model's response
    full_response: str = ""  # The complete raw response from the model

    # Evaluation field
    is_correct: bool = False  # Boolean indicating if the extracted answer is correct

    @classmethod
    def from_dict(cls, data: dict) -> "OutputStructure":
        """
        Create an OutputStructure instance from a dictionary.

        Args:
            data: Dictionary containing output data, typically loaded from a JSON file.

        Returns:
            An OutputStructure instance populated with data.
        """
        return cls(
            datasource=data.get("datasource", ""),
            video_id=data.get("video_id", ""),
            prompt=data.get("prompt", ""),
            correct_answer=data.get("correct_answer", ""),
            reasoning=data.get("reasoning", ""),
            answer=data.get("answer", ""),
            full_response=data.get("full_response", ""),
            is_correct=data.get("is_correct", False),
            output_json_fname=data.get(
                "output_json_fname"
            ),  # Note: output_json_fname is used internally but not saved in the final JSON
        )


# === Response Parsing Functions ===

# Regex pattern to find content within <answer> tags (non-greedy)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# Regex pattern to find content within <think> tags (non-greedy)
REASONING_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Regex pattern to find a single uppercase letter (A-Z)
SINGLE_LETTER_PATTERN = re.compile(r"[A-Z]")


def parse_reasoning_response(output_text: str) -> tuple[str, str]:
    """
    Parses model output text expected to contain <think> and <answer> tags.

    Extracts the single letter answer (A-Z) from within the <answer> tag
    and the reasoning from within the <think> tag.

    Args:
        output_text: The raw text response from the model.

    Returns:
        A tuple containing the extracted answer (str) and reasoning (str).
        Returns empty strings if patterns are not found.
    """
    answer = ""
    reasoning = ""

    # Find the <answer> tag content
    answer_match = ANSWER_PATTERN.search(output_text)
    if answer_match:
        # Look for a single uppercase letter within the extracted answer content
        letter_match = SINGLE_LETTER_PATTERN.search(answer_match.group(1))
        if letter_match:
            answer = letter_match.group(0)

    # Find the <think> tag content
    reasoning_match = REASONING_PATTERN.search(output_text)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return answer, reasoning


def parse_letter_response(output_text: str) -> tuple[str, str]:
    """
    Parses model output text expected to contain a single letter answer (A-Z).

    Extracts the first single uppercase letter found in the text.
    Reasoning is not extracted by this parser.

    Args:
        output_text: The raw text response from the model.

    Returns:
        A tuple containing the extracted answer (str) and an empty reasoning string (str).
        Returns empty strings if no uppercase letter is found.
    """
    answer = ""
    reasoning = ""  # This parser does not extract reasoning

    # Look for the first single uppercase letter in the text
    letter_match = SINGLE_LETTER_PATTERN.search(output_text)
    if letter_match:
        answer = letter_match.group(0)

    return answer, reasoning


def save_single_file(args: tuple[str, list[OutputStructure]]):
    """
    Helper function to save a list of OutputStructure objects to a single JSON file.
    Intended for use with multiprocessing.

    Args:
        args: A tuple containing the target output JSON filename (str)
              and a list of OutputStructure objects to save to that file.
    """
    output_json_fname, results = args

    # Convert OutputStructure objects to dictionaries for JSON serialization
    results_dicts = [attrs.asdict(result) for result in results]

    # Remove the internal 'output_json_fname' key from dictionaries before saving
    for result_dict in results_dicts:
        result_dict.pop("output_json_fname", None)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_json_fname)
    if output_dir:  # Only try to create directory if path is not just a filename
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_json_fname, "w") as f:
            json.dump(results_dicts, f, indent=4)
        log.info(f"Successfully wrote {len(results)} results to '{output_json_fname}'")
    except Exception as e:
        log.error(
            f"An error occurred while writing to JSON file '{output_json_fname}': {e}"
        )


def save_results_parallel(
    output_results: list[OutputStructure], num_processes: int = 4
):
    """
    Saves a list of OutputStructure objects to their respective output files
    using multiprocessing for parallel processing.

    Groups results by their 'output_json_fname' attribute before saving.

    Args:
        output_results: A list of OutputStructure objects to save.
        num_processes: The number of worker processes to use for parallel saving.
                       Defaults to 4.
    """
    # Group output results by their designated output file name
    output_files: dict[str, list[OutputStructure]] = {}

    for result in output_results:
        if result.output_json_fname not in output_files:
            output_files[result.output_json_fname] = []
        output_files[result.output_json_fname].append(result)

    log.info(
        f"Saving results to {len(output_files)} output files using {num_processes} processes..."
    )

    # Prepare arguments for the save_single_file helper function
    save_args = [
        (output_json_fname, results)
        for output_json_fname, results in output_files.items()
    ]

    # Use ProcessPoolExecutor to save files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit each file-saving task to the executor
        futures = {executor.submit(save_single_file, arg): arg for arg in save_args}

        # Wait for tasks to complete and report results or errors
        for future in concurrent.futures.as_completed(futures):
            arg = futures[future]
            try:
                future.result()  # Retrieve result (or exception)
            except Exception as e:
                log.error(
                    f"Error occurred during parallel saving for file {arg[0]}: {e}"
                )

    log.info("Parallel saving process completed.")
