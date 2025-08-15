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
import logging as log
import os
from functools import partial
from typing import Any

import attrs
from qwen_vl_utils import process_vision_info

from tools.eval.utils.output import OutputStructure


#  ------------- input -------------
@attrs.define(slots=False)
class InputStructure:
    """
    Represents the input structure for a single video-based question task.

    Attributes:
        datasource: The name of the dataset or datasource (e.g., "av_meta_actions_20250227").
        video_id: The identifier for the video.
        question: The user's question about the video.
        index2ans: A dictionary mapping answer indices (e.g., "A", "B") to answer text.
        question_idx: The index of the question within a larger set of questions.
        correct_answer: The correct answer index (uppercase, e.g., "A").
        video_cache_path: Optional path to the locally cached video file. Defaults to None.
        prompt: Optional formatted prompt string or structure for the model. Defaults to None.
    """

    datasource: str
    video_id: str
    question: str
    index2ans: dict[str, str]
    question_idx: int
    correct_answer: str
    video_cache_path: str | None = None
    prompt: str | list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(
        cls, datasource: str, video_id: str, data: dict[str, Any], idx: int
    ) -> "InputStructure":
        """
        Create an InputStructure instance from a dictionary containing question data.

        Args:
            datasource: The name of the datasource.
            video_id: The ID of the video associated with the data.
            data: A dictionary containing the question, answer options, and correct answer.
            idx: The index of the question.

        Returns:
            An InputStructure instance populated with the provided data.
        """
        # Handle potential variations in the key for the correct answer
        answer = data.get("answer", data.get("correct_answer"))
        return cls(
            datasource=datasource,
            video_id=video_id,
            question=data.get("question", ""),  # Provide default empty string
            index2ans=data.get("index2ans", {}),  # Provide default empty dict
            question_idx=idx,
            correct_answer=answer,
        )


def load_datasource_list(datasource_file: str) -> list[str]:
    """
    Loads a list of datasource names from a text file.

    The file is expected to contain one datasource name per line. Empty lines
    and leading/trailing whitespace are ignored.

    Args:
        datasource_file: The path to the text file containing the list of datasources.

    Returns:
        A list of datasource names. Returns an empty list if the file is not found
        or an error occurs during reading.
    """
    datasource_list: list[str] = []
    try:
        with open(datasource_file) as f:
            # Read lines, remove whitespace, and filter out empty lines
            datasource_list = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        log.error(f"Error: Datasource file not found at '{datasource_file}'")
    except Exception as e:
        log.error(f"An error occurred while reading '{datasource_file}': {e}")

    return datasource_list


def get_video_path(datasource: str, video_id: str, data_dir: str) -> str:
    """
    Constructs the expected path to a video file based on datasource, video ID, and data directory.

    Includes special handling for specific datasources that require a file suffix.
    Logs an error if the constructed path does not exist.

    Args:
        datasource: The name of the datasource.
        video_id: The ID of the video.
        data_dir: The root directory where datasource data is stored.

    Returns:
        The full path to the video file. Note: This function does not guarantee
        the file exists, it only constructs the path.
    """
    # Special suffix handling based on datasource name
    v_suffix = (
        ".camera_front_wide_120fov" if "av_meta_actions_20250227" in datasource else ""
    )

    # Define the expected video file path
    video_path = os.path.join(
        data_dir, datasource, "clips", f"{video_id}{v_suffix}.mp4"
    )

    # Log a warning if the file does not exist at the constructed path
    if not os.path.exists(video_path):
        log.warning(
            f"Video file not found at expected path: {video_path}. "
            f"Please ensure the video is downloaded to the correct location within '{data_dir}'."
        )

    return video_path


def get_prompt(
    input_question: InputStructure, answer_type: str
) -> list[dict[str, Any]]:
    """
    Generates the chat prompt structure for the model based on the question and answer type.

    Formats the question and answer choices into a prompt string and wraps it
    in a system/user chat format. Adds specific instructions based on `answer_type`.

    Args:
        input_question: The InputStructure object containing question details.
        answer_type: The desired format for the answer ("letter" or "reasoning").

    Returns:
        A list of dictionaries representing the chat history prompt structure
        (system and user messages).
    """
    system_prompt = "Answer the questions."
    question = input_question.question
    choices = input_question.index2ans

    # Format question with choices
    prompt_text = question + "\n"
    prompt_text += "\n".join([f"({i}) {choice}" for i, choice in choices.items()])
    prompt_text += "\nAnswer with the option's letter from the given choices directly."

    # Add reasoning instructions if required
    if answer_type == "reasoning":
        prompt_text += "\nPlease answer the question in the following format: <think> your reasoning </think> <answer> your single letter answer </answer>."

    # Structure the prompt as a chat history
    prompt: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": prompt_text,
        },
    ]
    return prompt


def get_video_path_and_prompt(
    video_input_task: InputStructure,
    data_dir: str,
    answer_type: str,
) -> InputStructure:
    """
    Worker function executed in parallel to determine the video path and generate
    the prompt for a single input task.

    Updates the input task object with the video cache path and the generated prompt.

    Args:
        video_input_task: The InputStructure object for the task.
        data_dir: The root directory where video data is stored.
        data_cache_dir: Path to the data cache directory (currently unused).
        answer_type: The desired format for the answer ("letter" or "reasoning").
        use_hf_dataset: Flag indicating if Hugging Face dataset is used (currently unused).

    Returns:
        The updated InputStructure object with `video_cache_path` and `prompt` fields populated.
    """
    # Get the video file path
    video_path = get_video_path(
        video_input_task.datasource, video_input_task.video_id, data_dir
    )

    # Generate the prompt for the task
    prompt = get_prompt(video_input_task, answer_type)

    # Update the input task object
    video_input_task.prompt = prompt
    video_input_task.video_cache_path = video_path

    log.debug(
        f"Processed video path and prompt for task: video_id={video_input_task.video_id}, "
        f"datasource={video_input_task.datasource}, question_idx={video_input_task.question_idx}"
    )

    return video_input_task


def load_videos_and_prompts_parallel(
    input_tasks: list[InputStructure],
    data_dir: str,
    answer_type: str,
    num_processes: int,
) -> list[InputStructure]:
    """
    Loads video paths and generates prompts for a list of input tasks in parallel.

    Uses a ProcessPoolExecutor to distribute the work of `get_video_path_and_prompt`
    across multiple processes. Handles collecting results and maintaining the
    original order of tasks. Falls back to sequential processing if parallel
    execution is not possible or beneficial (e.g., zero tasks or workers).

    Args:
        input_tasks: A list of InputStructure objects.
        data_dir: The root directory where video data is stored.
        answer_type: The desired format for the answer ("letter" or "reasoning").
        num_processes: The maximum number of processes to use for parallel execution.

    Returns:
        A list of updated InputStructure objects with `video_cache_path` and
        `prompt` fields populated. The order of tasks in the output list
        corresponds to the order in the input list.
    """
    updated_input_tasks: list[InputStructure] = []
    num_workers = min(num_processes, len(input_tasks))

    if num_workers > 0 and len(input_tasks) > 0:
        log.info(
            f"Loading video paths and generating prompts in parallel for {len(input_tasks)} tasks "
            f"using {num_workers} processes."
        )

        # Create a partial function with fixed arguments for the worker
        worker_fn = partial(
            get_video_path_and_prompt,
            data_dir=data_dir,
            answer_type=answer_type,
        )

        # Use ProcessPoolExecutor for CPU-bound tasks (file system operations)
        # Store results with their original index to sort later
        results_with_index: list[tuple[int, InputStructure]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            # Submit tasks and map futures back to their original index
            future_to_idx = {
                executor.submit(worker_fn, task): i
                for i, task in enumerate(input_tasks)
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    updated_task = future.result()
                    results_with_index.append((idx, updated_task))
                except Exception as e:
                    log.error(
                        f"Error processing task {idx} for video path and prompt: {e}. "
                        f"Keeping original task data."
                    )
                    # Append the original task in case of processing error
                    results_with_index.append((idx, input_tasks[idx]))

        # Sort results by original index to restore order
        results_with_index.sort(key=lambda x: x[0])
        updated_input_tasks = [task for idx, task in results_with_index]

    else:
        # Fallback to sequential processing
        log.info("Using sequential processing for video loading and prompt generation.")
        for task in input_tasks:
            try:
                updated_task = get_video_path_and_prompt(
                    task,
                    data_dir,
                    answer_type,
                )
                updated_input_tasks.append(updated_task)
            except Exception as e:
                log.error(
                    f"Error processing task sequentially for video path and prompt: {e}. "
                    f"Skipping task: video_id={task.video_id}, question_idx={task.question_idx}"
                )
                # Optionally, you might append the original task or None here depending on desired behavior on error
                pass  # Currently, errors in sequential mode cause the task to be skipped from the output list

    return updated_input_tasks


def check_file_exists(output_result: OutputStructure) -> bool:
    """
    Checks if the output file specified in the OutputStructure object exists on the local filesystem.

    Args:
        output_result: An OutputStructure object containing the path to the output file.

    Returns:
        True if the file exists, False otherwise.
    """
    # Use os.path.exists to check for file existence
    exists = os.path.exists(output_result.output_json_fname)

    if not exists:
        log.debug(f"Output file does not exist: {output_result.output_json_fname}")
    else:
        log.debug(f"Output file exists: {output_result.output_json_fname}")

    return exists


def skip_saved_results(
    input_tasks: list[InputStructure],
    output_results: list[OutputStructure],
    num_processes: int = 40,
) -> tuple[list[InputStructure], list[OutputStructure]]:
    """
    Filters out tasks for which a corresponding output file already exists.

    Checks for the existence of output files in parallel using a ProcessPoolExecutor.
    Returns new lists containing only the input tasks and output results for
    which the output file was not found, preserving the original order.

    Args:
        input_tasks: A list of InputStructure objects.
        output_results: A list of OutputStructure objects, expected to be
                        one-to-one correspondence with `input_tasks`.
        num_processes: The maximum number of processes to use for parallel file checking.

    Returns:
        A tuple containing two lists: (filtered_input_tasks, filtered_output_results).
        These lists contain only the tasks and results for which the output file
        did not exist.
    """
    if not input_tasks or not output_results:
        log.info("No tasks or output results to process for skipping.")
        return [], []

    if len(input_tasks) != len(output_results):
        log.error(
            f"Mismatch between number of input tasks ({len(input_tasks)}) "
            f"and output results ({len(output_results)}). Skipping all tasks."
        )
        return [], []  # Return empty lists if there's a mismatch

    log.info(
        f"Checking for existing results for {len(input_tasks)} tasks "
        f"using {min(num_processes, len(input_tasks))} processes."
    )

    # Use ProcessPoolExecutor for parallel file existence checks (I/O bound)
    exists_by_idx: dict[int, bool] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit checks and map futures to their original index
        future_to_idx = {
            executor.submit(check_file_exists, output_result): i
            for i, output_result in enumerate(output_results)
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                exists = future.result()
                exists_by_idx[idx] = exists
            except Exception as e:
                log.error(
                    f"Error checking file existence for task {idx}: {e}. "
                    f"Assuming file does NOT exist."
                )
                # Assume file doesn't exist in case of error to re-process the task
                exists_by_idx[idx] = False

    # Filter tasks based on the existence check results, maintaining order
    new_input_tasks: list[InputStructure] = []
    new_output_results: list[OutputStructure] = []
    skipped_count = 0
    for i in range(len(input_tasks)):
        if not exists_by_idx.get(i, False):  # Default to False if index is missing
            new_input_tasks.append(input_tasks[i])
            new_output_results.append(output_results[i])
        else:
            skipped_count += 1

    log.info(f"Skipped {skipped_count} tasks with existing results.")
    log.info(f"Proceeding with {len(new_input_tasks)} tasks.")

    return new_input_tasks, new_output_results


def prepare_single_model_input(
    input_task: InputStructure, processor: Any, fps: int
) -> Any | None:
    """
    Worker function to prepare the input data for a single model inference task.

    Integrates the video path into the prompt structure and applies the model's
    chat template. Uses `process_vision_info` and formats the final model input
    based on whether a Hugging Face checkpoint is being used.

    Args:
        input_task: The InputStructure object for the task.
        processor: The model's processor or tokenizer object, used for template application
                   and input formatting.
        fps: The frames per second to associate with the video input.

    Returns:
        The prepared model input object or None
        if an error occurs during preparation.
    """
    try:
        # Ensure the prompt is in the expected list format
        if not isinstance(input_task.prompt, list) or not input_task.prompt:
            log.error(
                f"Invalid prompt format for task: video_id={input_task.video_id}, "
                f"question_idx={input_task.question_idx}. Expected list."
            )
            return None

        # Add video information to the user message content
        # Assuming the user message is the second element and its content is text
        if (
            len(input_task.prompt) > 1
            and input_task.prompt[1]["role"] == "user"
            and isinstance(input_task.prompt[1]["content"], str)
        ):
            input_task.prompt[1]["content"] = [
                {"type": "video", "video": input_task.video_cache_path, "fps": fps},
                {"type": "text", "text": input_task.prompt[1]["content"]},
            ]
        elif (
            len(input_task.prompt) > 1
            and input_task.prompt[1]["role"] == "user"
            and isinstance(input_task.prompt[1]["content"], list)
        ):
            # If content is already a list, prepend video info
            input_task.prompt[1]["content"].insert(
                0, {"type": "video", "video": input_task.video_cache_path, "fps": fps}
            )
        else:
            log.error(
                f"Could not add video info to prompt for task: video_id={input_task.video_id}, question_idx={input_task.question_idx}. Unexpected prompt structure."
            )
            return None

        # Apply the model's chat template to get the final text prompt
        # Use add_generation_prompt=True to include the prompt part that signals
        # the model to start generating the response.
        processed_text_prompt = processor.apply_chat_template(
            input_task.prompt, tokenize=False, add_generation_prompt=True
        )

        # Process vision information (image/video paths) from the prompt structure
        image_inputs, video_inputs = process_vision_info(input_task.prompt)

        # Format the final model input based on the target model type (HF vs. custom)

        # Format for a potential custom model structure
        # Assuming video_inputs contains a list and we need the first item
        if not video_inputs:
            log.error(
                f"No video inputs found for task: video_id={input_task.video_id}, question_idx={input_task.question_idx}. Cannot prepare model input."
            )
            return None

        model_input = {
            "prompt": processed_text_prompt,
            "multi_modal_data": {
                "video": [video_inputs[0]]
            },  # Assuming first video input is relevant
        }

        log.debug(
            f"Prepared model input for task: video_id={input_task.video_id}, "
            f"question_idx={input_task.question_idx}"
        )
        return model_input
    except Exception as e:
        log.error(
            f"Error preparing model input for task: video_id={input_task.video_id}, "
            f"question_idx={input_task.question_idx}: {e}"
        )
        return None  # Return None to indicate failure


def prepare_model_inputs_parallel(
    input_tasks: list[InputStructure],
    processor: Any,
    num_processes: int = 40,
    fps: int = 4,
) -> list[Any]:
    """
    Prepares model inputs for a list of input tasks in parallel using threads.

    This function is suitable for I/O-bound tasks like reading files or
    processing data structures, as it uses ThreadPoolExecutor. It calls
    `prepare_single_model_input` for each task. Handles collecting results
    and filters out any tasks that failed during preparation.

    Args:
        input_tasks: A list of InputStructure objects to prepare inputs for.
        processor: The model's processor/tokenizer object.
        num_processes: The maximum number of threads to use for parallel execution.
        fps: The frames per second to associate with video inputs.

    Returns:
        A list of prepared model input objects. Tasks that failed during
        preparation are excluded from the list. The order of inputs
        corresponds to the order of successful tasks in the input list.
    """
    if not input_tasks:
        log.info("No input tasks to prepare model inputs for.")
        return []

    # Determine the number of workers, up to the number of tasks
    num_workers = min(num_processes, len(input_tasks))

    log.info(
        f"Preparing model inputs in parallel for {len(input_tasks)} tasks "
        f"using {num_workers} threads."
    )

    # Create a partial function with fixed arguments for the worker
    worker_fn = partial(
        prepare_single_model_input,
        processor=processor,
        fps=fps,
    )

    processed_inputs_with_index: list[tuple[int, Any]] = []

    # Use ThreadPoolExecutor for potentially I/O-bound model input preparation
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks and map futures to their original index
        future_to_idx = {
            executor.submit(worker_fn, task): i for i, task in enumerate(input_tasks)
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                model_input = future.result()
                if model_input is not None:  # Only append successful results
                    processed_inputs_with_index.append((idx, model_input))
            except Exception as e:
                # This catch might be redundant if prepare_single_model_input handles errors,
                # but included for robustness.
                log.error(
                    f"Unexpected error preparing model input for task {idx}: {e}. Skipping task."
                )
                pass  # Skip this task if an unexpected error occurs

    # Sort successful results by original index and extract the inputs
    processed_inputs_with_index.sort(key=lambda x: x[0])
    inputs = [inp for idx, inp in processed_inputs_with_index]

    if len(inputs) < len(input_tasks):
        log.warning(
            f"Successfully prepared inputs for {len(inputs)} out of {len(input_tasks)} tasks."
        )

    return inputs
