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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cosmos-reason1-benchmark",
#   "cosmos-reason1-utils",
# ]
# [tool.uv]
# exclude-newer = "2025-08-05T00:00:00Z"
# [tool.uv.sources]
# cosmos-reason1-benchmark = { path = "../../", editable = true }
# cosmos-reason1-utils = { path = "../../../../cosmos_reason1_utils", editable = true }
# ///

"""Evaluate a model on a dataset."""
# ruff: noqa: E402

from cosmos_reason1_utils.script import init_script

init_script()

import json
import logging as log
import os
import time
from argparse import ArgumentParser
from typing import Any

import yaml
from PIL import Image

# Increase maximum allowed pixels for Pillow images to handle large inputs
Image.MAX_IMAGE_PIXELS = 933120000
# Enable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configure basic logging
log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Check if debug model is enabled via environment variable
DEBUG_MODEL: bool = os.getenv("DEBUG_MODEL", "0") == "1"

# Import model and tokenizer based on DEBUG_MODEL flag
if DEBUG_MODEL:
    from tools.eval_utils.dummy_model import DummyModel, DummyTokenizer, SamplingParams

    # Define type aliases for clarity when using the dummy model
    LLM = DummyModel
    Processor = DummyTokenizer
else:
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    # Define type alias for clarity when using the real processor
    Processor = AutoProcessor

# Import custom evaluation utilities
from tools.eval.utils.input import (
    InputStructure,
    load_videos_and_prompts_parallel,
    prepare_model_inputs_parallel,
    skip_saved_results,
)
from tools.eval.utils.model_download import download_checkpoint, download_tokenizer
from tools.eval.utils.output import (
    OutputStructure,
    parse_letter_response,
    parse_reasoning_response,
    save_results_parallel,
)


# === Model Definition Functions ===
def define_model(
    tokenizer_model_name: str,
    model_name: str,
    dtype: str,
    tp_size: int | None,
    max_length: int = 12800,
) -> tuple[LLM, Processor]:
    """
    Defines and loads the language model and its processor.

    Args:
        tokenizer_name: The name of the tokenizer (e.g., "qwen2.5-vl-7b").
        model_name: Name of the model.
        dtype: Data type for model weights ("bfloat16" or "float16").
        tp_size: Tensor parallel size for VLLM, or None to use defaults.
        max_length: Maximum sequence length for the model.

    Returns:
        A tuple containing the loaded model and its processor.
    """

    hf_cache_dir = os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    )
    checkpoint_output_dir = os.path.join(hf_cache_dir, model_name)

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_output_dir, exist_ok=True)

    # Download checkpoint if not already present
    download_checkpoint(model_name, checkpoint_output_dir)

    # Download tokenizer if not already present
    download_tokenizer(tokenizer_model_name, checkpoint_output_dir)

    log.info("Using VLLM backend.")
    # Allow longer max model length in VLLM
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    # Initialize VLLM LLM object
    llm = LLM(
        model=checkpoint_output_dir,
        tokenizer=checkpoint_output_dir,
        trust_remote_code=True,
        dtype=dtype,
        # Specify multimedia limit per prompt (e.g., one video, zero images)
        limit_mm_per_prompt={"video": 1, "image": 0},
        tensor_parallel_size=tp_size,
        # Capture sequence length for KV cache optimization (optional)
        max_seq_len_to_capture=16384,
        max_model_len=max_length,
    )

    # Load processor from the same checkpoint directory
    processor: Processor = AutoProcessor.from_pretrained(
        checkpoint_output_dir, max_length=max_length
    )

    return llm, processor


# === Task Generation Functions ===


def make_tasks_from_single_video(
    output_json_fname: str,
    qa_pairs: list[dict[str, Any]],
    video_id: str,
    datasource_name: str,
) -> tuple[list[InputStructure], list[OutputStructure]]:
    """
    Creates InputStructure and OutputStructure objects for all questions related to a single video.

    Args:
        output_json_fname: The base filename for saving results for this video.
        qa_pairs: A list of question-answer dictionaries for the video.
        video_id: The unique identifier for the video.
        datasource_name: The name of the dataset the video belongs to.

    Returns:
        A tuple containing:
        - A list of InputStructure objects for questions needing evaluation.
        - A list of OutputStructure objects to store evaluation results.
    """
    input_questions: list[InputStructure] = []
    output_results: list[OutputStructure] = []

    for idx, qa in enumerate(qa_pairs):
        # Create InputStructure for the current question
        input_questions.append(
            InputStructure.from_dict(datasource_name, video_id, qa, idx)
        )
        # Create corresponding OutputStructure to store results
        output_results.append(
            OutputStructure(
                datasource=datasource_name,
                video_id=video_id,
                # Get the correct answer (handles variations in dict key)
                correct_answer=qa.get("correct_answer", qa.get("answer", "")),
                output_json_fname=output_json_fname,
                prompt="",  # This will be filled later
            )
        )

    return input_questions, output_results


def make_all_tasks(
    datasource_list: list,
    results_output_folder: str,
    data_dir: str,
    limit: int = -1,
) -> tuple[list[InputStructure], list[OutputStructure]]:
    """
    Gathers all evaluation tasks from the specified datasources. Supports loading from
    a list of datasource names (for S3 paths) or a Hugging Face dataset.

    Args:
        datasource_list: List of datasource names.
        results_output_folder: Base directory to save evaluation results.
        limit: Maximum number of tasks to gather across all datasources (-1 for no limit).

    Returns:
        A tuple containing:
        - A list of all InputStructure objects to be evaluated.
        - A list of all corresponding OutputStructure objects.
    """

    input_tasks: list[InputStructure] = []  # Stores all input tasks
    output_results: list[OutputStructure] = []  # Stores all output result objects

    # Process each datasource
    for datasource_name in datasource_list:
        log.info(f"Gathering tasks from dataset: {datasource_name}")

        curr_tasks: list[InputStructure] = []
        curr_results: list[OutputStructure] = []

        # Construct the video path properly
        video_path = os.path.join(data_dir, datasource_name, "clips")

        # Check if video_path exists
        if not os.path.exists(video_path):
            log.error(f"Video path does not exist: {video_path}")
            continue

        # Get video files
        try:
            video_files = [
                f
                for f in os.listdir(video_path)
                if os.path.isfile(os.path.join(video_path, f))
            ]
        except OSError as e:
            log.error(f"Error reading video directory {video_path}: {e}")
            continue

        # Load QA pairs properly
        try:
            qa_pairs_path = os.path.join(
                data_dir, datasource_name, f"{datasource_name}_benchmark_qa_pairs.json"
            )  # Assuming JSON file name
            with open(qa_pairs_path) as f:
                qa_pairs = json.load(f)

            # Convert qa_pairs list to a dictionary
            qa_pairs_dict = {}
            for item in qa_pairs:
                video_path = item.get("video", "")
                if video_path.startswith("clips/") and "." in video_path:
                    video_id = video_path.split("/")[-1].rsplit(".", 1)[0]
                    qa_data = item.get("qa_pairs")
                    if isinstance(qa_data, list):
                        qa_pairs_dict[video_id] = qa_data
                    elif isinstance(qa_data, dict):
                        qa_pairs_dict[video_id] = [qa_data]

        except (json.JSONDecodeError, FileNotFoundError) as e:
            log.error(f"Error loading QA pairs for {datasource_name}: {e}")
            continue

        # Process each item (video) in the dataset split
        for video_file in sorted(video_files):
            # Verify video file extension
            if not video_file.endswith(
                (".mp4", ".avi", ".mov")
            ):  # Add relevant video extensions
                continue

            video_id = video_file.rsplit(".", 1)[0]

            # Check if video_id exists in qa_pairs
            if video_id not in qa_pairs_dict:
                log.warning(f"No QA pairs found for video ID: {video_id}")
                continue
            video_qa_pairs = qa_pairs_dict.get(video_id)
            if video_qa_pairs is None:
                log.warning(f"No QA pairs found for video ID: {video_id}")
                continue
            # Define the output JSON file path for this video
            output_json_fname = os.path.join(
                results_output_folder, datasource_name, f"{video_id}.json"
            )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_json_fname), exist_ok=True)

            log.info(
                f"Processing video: {datasource_name}/{video_id}, results will be saved to {output_json_fname}"
            )

            # Create tasks and results for the current video's questions
            try:
                video_input_tasks, video_output_results = make_tasks_from_single_video(
                    output_json_fname, video_qa_pairs, video_id, datasource_name
                )
            except Exception as e:
                log.error(f"Error processing video {video_id}: {e}")
                continue

            # Add tasks and outputs for current video to the list for this datasource
            curr_tasks.extend(video_input_tasks)
            curr_results.extend(video_output_results)

            # Check if the task limit has been reached for this datasource
            if limit > 0 and len(curr_tasks) >= limit:
                log.info(
                    f"Limit ({limit}) reached for datasource '{datasource_name}'. Stopping."
                )
                break  # Stop processing videos for this datasource

        # Add tasks and results from the current datasource to the overall lists
        input_tasks.extend(curr_tasks)
        output_results.extend(curr_results)

    return input_tasks, output_results


# === Model Execution Functions ===


def run_model(
    model: LLM,
    inputs: list[str],  # VLLM generate takes list of prompts
    input_tasks: list[InputStructure],
    output_results: list[OutputStructure],
    stop_token_id: int,
    answer_type: str,
    max_retries: int = 3,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    seed: int = 0,
) -> None:
    """
    Runs the VLLM model on the provided inputs and processes the outputs.
    Includes retry logic for tasks resulting in empty answers.

    Args:
        model: The loaded VLLM model.
        inputs: A list of prompt strings for the model.
        input_tasks: List of original InputStructure objects.
        output_results: List of OutputStructure objects to update with results.
        stop_token_id: The token ID to stop generation at.
        answer_type: Expected format of the answer ("letter" or "reasoning").
        max_retries: Maximum number of times to retry tasks with empty answers.
        max_tokens: Maximum number of tokens to generate per task.
        temperature: Sampling temperature.
        repetition_penalty: Penalty for repeating tokens.
        presence_penalty: Penalty for using tokens already present.
        frequency_penalty: Penalty based on token frequency.
        seed: Random seed for sampling.
    """
    # Configure sampling parameters based on the expected answer type
    if answer_type == "letter":
        # Use greedy decoding (temperature=0, top_k=1) for letter answers
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,  # Generate only a few tokens for a letter answer
            stop_token_ids=[stop_token_id],
            top_k=1,
            seed=seed,
        )
    else:  # answer_type == "reasoning"
        # Use specified sampling parameters for reasoning answers
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,  # Use top_p sampling
            stop_token_ids=[stop_token_id],
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
        )

    log.info(f"Generating outputs for {len(inputs)} tasks using VLLM...")
    # Generate outputs for all inputs in a single batch
    list_of_requestoutput = model.generate(inputs, sampling_params)
    log.info(
        f"Finished VLLM generation. Received {len(list_of_requestoutput)} outputs."
    )

    empty_answer_indices: list[
        int
    ] = []  # Keep track of indices for tasks with empty answers

    # Process the initial generation outputs
    for i, (requestoutput, video_input_task, video_output_result) in enumerate(
        zip(list_of_requestoutput, input_tasks, output_results, strict=False)
    ):
        output_text = requestoutput.outputs[0].text  # Get the generated text
        # Parse the generated text based on the expected answer type
        if answer_type == "letter":
            answer, reasoning = parse_letter_response(output_text)
        else:
            answer, reasoning = parse_reasoning_response(output_text)

        # Store the generated results in the OutputStructure object
        video_output_result.prompt = (
            video_input_task.prompt
        )  # Store the original prompt
        video_output_result.reasoning = reasoning
        video_output_result.answer = answer
        video_output_result.full_response = output_text
        # Check if the generated answer is correct (case-insensitive comparison)
        video_output_result.is_correct = (
            answer.lower() == video_input_task.correct_answer.lower()
        )

        # If the generated answer is empty, add its index to the retry list
        if not answer:
            empty_answer_indices.append(i)

    # --- Retry logic for empty answers ---
    current_empty_indices = empty_answer_indices  # Initialize retry list
    for retry_count in range(max_retries):
        if not current_empty_indices:
            log.info("No more empty answers. Retries finished.")
            break  # Exit retry loop if no empty answers remain

        log.info(
            f"Found {len(current_empty_indices)} empty answers. Retrying batch ({retry_count + 1}/{max_retries})..."
        )

        # Prepare inputs specifically for the tasks that had empty answers
        retry_inputs: list[str] = [inputs[i] for i in current_empty_indices]

        # Adjust sampling parameters for retries (increase max tokens and temperature)
        # This encourages the model to generate different responses
        sampling_params.max_tokens = max_tokens + 256 * (retry_count + 1)
        sampling_params.temperature = min(temperature + (0.05 * (retry_count + 1)), 0.9)
        log.info(
            f"  Retry {retry_count + 1} sampling params: max_tokens={sampling_params.max_tokens}, temperature={sampling_params.temperature}"
        )

        # Generate outputs for the retry batch
        retry_outputs = model.generate(retry_inputs, sampling_params)

        still_empty_indices: list[
            int
        ] = []  # List to hold indices that are still empty after this retry
        # Process the retry outputs
        for batch_idx, original_idx in enumerate(current_empty_indices):
            retry_text = (
                retry_outputs[batch_idx].outputs[0].text
            )  # Get the generated text from retry
            # Parse the generated text
            if answer_type == "letter":
                answer, reasoning = parse_letter_response(retry_text)
            else:
                answer, reasoning = parse_reasoning_response(retry_text)

            # If a valid answer was obtained in this retry, update the result
            if answer:
                output_results[original_idx].reasoning = reasoning
                output_results[original_idx].answer = answer
                output_results[original_idx].full_response = retry_text
                # Update correctness based on the new answer
                output_results[original_idx].is_correct = (
                    answer.lower() == input_tasks[original_idx].correct_answer.lower()
                )
            else:
                # If still no answer, add to the list for the next retry
                still_empty_indices.append(original_idx)

        # Update the list of indices to retry for the next iteration
        current_empty_indices = still_empty_indices

    # Report any tasks that still have empty answers after all retries
    if current_empty_indices:
        log.warning(
            f"{len(current_empty_indices)} tasks still have empty answers after {max_retries} retries."
        )
        # Log details of tasks that failed to produce an answer
        for original_idx in current_empty_indices:
            log.warning(
                f"  - Task from video '{input_tasks[original_idx].video_id}' in datasource '{input_tasks[original_idx].datasource}' could not generate a valid answer."
            )


# === Main Function and Script Entry Point ===
def main():
    """
    Main function to set up evaluation, run the model, and save results.
    """
    # --- Argument Parsing ---
    parser = ArgumentParser(description="Run video language model evaluation.")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to YAML configuration file with model and evaluation parameters.",
    )

    # These arguments will remain as direct command-line options
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Base directory to save evaluation results.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing video data. Can be local or S3 path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit the total number of tasks to evaluate. Useful for debugging.",
    )
    parser.add_argument(
        "--skip_saved",
        action="store_true",
        help="Skip evaluating tasks for which results are already saved.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model to use for evaluation.",
    )

    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        with open(args.config_file) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        log.error(f"Error loading config file: {e}")
        return

    # Extract configuration sections
    model_config = config.get("model", {})
    eval_config = config.get("evaluation", {})
    gen_config = config.get("generation", {})

    # Retrieve datasets from config
    datasets = config.get("datasets")
    if not datasets:
        log.error("No datasets provided in config file")
        return

    # Convert datasets list to a temporary datasource file or use in-memory list
    # depending on what make_all_tasks expects
    if isinstance(datasets, list):
        log.info(f"Using datasets from config: {', '.join(datasets)}")
    else:
        log.error("'datasets' must be a list in the config file")
        return

    # --- Model Configuration ---
    model_name = args.model_name or model_config.get("model_name", None)
    tokenizer_model_name = model_config.get("tokenizer_model_name", "qwen2.5-vl-7b")
    dtype = model_config.get("dtype", "bfloat16")
    tp_size = model_config.get("tp_size", 1)
    max_length = model_config.get("max_length", 128000)

    # --- Evaluation Parameters ---
    answer_type = eval_config.get("answer_type", "reasoning")
    num_processes = eval_config.get("num_processes", 80)
    fps = eval_config.get("fps", 4)
    seed = eval_config.get("seed", 1)

    # --- Generation Parameters ---
    max_retries = gen_config.get("max_retries", 10)
    max_tokens = gen_config.get("max_tokens", 1024)
    temperature = gen_config.get("temperature", 0.6)
    repetition_penalty = gen_config.get("repetition_penalty", 1.0)
    presence_penalty = gen_config.get("presence_penalty", 0.0)
    frequency_penalty = gen_config.get("frequency_penalty", 0.0)

    # Append dtype and seed to results directory for better organization
    results_output_base = f"{args.results_dir}"
    log.info(f"Results base directory updated to: {results_output_base}")

    args.data_dir = f"{args.data_dir}/benchmark"

    # Log all effective arguments for reproducibility
    log.info("--- Script Configuration ---")
    log.info(f"  Config file: {args.config_file}")
    log.info(f"  Model: {model_name}")
    log.info(f"  Results directory: {args.results_dir}")
    log.info(f"  Data directory: {args.data_dir}")
    log.info(f"  Limit: {args.limit}")
    log.info("--- Model Configuration ---")
    log.info(f"  Tokenizer model name: {tokenizer_model_name}")
    log.info(f"  Data type: {dtype}")
    log.info(f"  Tensor parallel size: {tp_size}")
    log.info(f"  Max length: {max_length}")
    log.info("--- Evaluation Configuration ---")
    log.info(f"  Answer type: {answer_type}")
    log.info(f"  Number of processes: {num_processes}")
    log.info(f"  Skip saved: {args.skip_saved}")
    log.info(f"  FPS: {fps}")
    log.info(f"  Seed: {seed}")
    log.info("--- Generation Configuration ---")
    log.info(f"  Max retries: {max_retries}")
    log.info(f"  Max tokens: {max_tokens}")
    log.info(f"  Temperature: {temperature}")
    log.info(f"  Repetition penalty: {repetition_penalty}")
    log.info(f"  Presence penalty: {presence_penalty}")
    log.info(f"  Frequency penalty: {frequency_penalty}")
    log.info("------------------------")
    # Define the final output path structure
    # {results_dir_base}/{model_name}/{answer_type}/{datasource}/{video_id}.json
    results_output_dir = os.path.join(
        results_output_base, os.path.basename(model_name.rstrip("/")), answer_type
    )
    log.info(f"Evaluation results will be saved to: {results_output_dir}")

    # === Step 1: Gather all tasks across all datasources and videos ===
    log.info("Starting task gathering...")
    start_time = time.time()
    # make_all_tasks now takes datasets directly
    input_tasks, output_results = make_all_tasks(
        datasets,  # Use the datasets list directly instead of a file
        results_output_dir,  # Pass the full results directory
        args.data_dir,
        args.limit,
    )
    log.info(f"Initial number of tasks gathered: {len(input_tasks)}")

    # === Step 1.5: Skip saved results if requested ===
    if args.skip_saved:
        log.info("Checking for and skipping already saved results...")
        input_tasks, output_results = skip_saved_results(
            input_tasks, output_results, num_processes
        )
        log.info(f"Number of tasks remaining after skipping saved: {len(input_tasks)}")
        # If no tasks remain after skipping, exit gracefully
        if not input_tasks:
            log.info("All results are already saved or no tasks to evaluate. Exiting.")
            return  # Exit the main function
    log.info(f"Total tasks to evaluate after filtering: {len(input_tasks)}")
    log.info(f"Time taken to gather tasks: {time.time() - start_time:.2f} seconds")

    # === Step 2: Load videos and generate prompts in parallel ===
    log.info("Loading videos and generating prompts in parallel...")
    start_time = time.time()
    input_tasks = load_videos_and_prompts_parallel(
        input_tasks,
        data_dir=args.data_dir,  # Base directory for video data
        answer_type=answer_type,
        num_processes=num_processes,
    )
    log.info(
        f"Time taken to load videos and generate prompts: {time.time() - start_time:.2f} seconds"
    )

    # === Step 3: Load model and processor ===
    log.info("Loading model and processor...")
    start_time = time.time()
    if DEBUG_MODEL:
        log.warning("DEBUG_MODEL is enabled. Using DummyModel and DummyTokenizer.")
        model: LLM = DummyModel()
        processor: Processor = DummyTokenizer()
    else:
        # Define and load the actual model and processor
        model, processor = define_model(
            tokenizer_model_name,
            model_name,
            dtype,
            tp_size,
            max_length,
        )
    log.info(f"Time taken to load model: {time.time() - start_time:.2f} seconds")

    # === Step 4: Prepare model inputs ===
    log.info("Preparing model inputs in parallel...")
    start_time = time.time()
    # Prepare inputs based on the chosen backend (HF or VLLM)
    # This step tokenizes prompts and handles image/video encoding if needed
    inputs = prepare_model_inputs_parallel(
        input_tasks,
        processor,
        num_processes,
        fps,  # Pass FPS as it might be needed for input preparation
    )
    log.info(f"Prepared inputs for {len(inputs)} tasks.")
    log.info(
        f"Time taken to prepare model inputs: {time.time() - start_time:.2f} seconds"
    )

    # === Step 5: Generate outputs using the model ===
    log.info("Generating outputs using the model...")
    start_time = time.time()

    # Run evaluation using the VLLM backend
    # Need the EOS token ID from the processor's tokenizer for VLLM stopping
    run_model(
        model,
        inputs,  # VLLM expects list of strings (prompts) directly
        input_tasks,
        output_results,
        processor.tokenizer.eos_token_id,  # Pass EOS token ID for VLLM stopping
        answer_type,
        max_retries,
        max_tokens,
        temperature,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        seed,
    )
    log.info(
        f"Time taken for model generation and output processing: {time.time() - start_time:.2f} seconds"
    )

    # === Step 6: Save results in parallel ===
    log.info("Saving results in parallel...")
    start_time = time.time()
    # Save the updated OutputStructure objects to JSON files
    save_results_parallel(output_results, num_processes=num_processes)
    log.info(f"Time taken to save results: {time.time() - start_time:.2f} seconds")
    log.info("Evaluation completed.")


# Script entry point
if __name__ == "__main__":
    main()
