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
"""Base evaluator for Cosmos-RL."""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging as log
import concurrent.futures
from tqdm import tqdm

from cosmos_rl.utils.lora_utils import get_base_model_path, merge_lora_model, should_enable_lora
from cosmos_rl.utils.tao_status_logger import log_tao_status
from cosmos_rl.evaluation.utils.model_download import download_checkpoint, download_tokenizer
from qwen_vl_utils import process_vision_info

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
Processor = AutoProcessor

# Import status callback utility for progress updates
try:
    from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker_utils import send_progress_status_callback
    STATUS_CALLBACK_AVAILABLE = True
except ImportError:
    STATUS_CALLBACK_AVAILABLE = False
    log.warning("Status callback not available - progress updates will not be sent")

COMPONENT_NAME = "Cosmos-RL Evaluation"
class BaseEvaluator(ABC):
    """Base evaluator for Cosmos-RL."""
    def __init__(self, config: Dict[str, Any], enable_lora: bool = False) -> None:
        """Initialize the BaseEvaluator."""
        self.config = config
        self.enable_lora = enable_lora
        self.model = None
        self.processor = None
        self.model_config = config.get("model", {})
        self.eval_config = config.get("evaluation", {})
        self.gen_config = config.get("generation", {})
        self.vision_config = config.get("vision", {})
        self.dataset_cfg = config.get("dataset", {})

    def _send_status_callback(self, message: str) -> None:
        """Send status callback to prevent timeout.

        Args:
            message: Status message to send
        """
        if STATUS_CALLBACK_AVAILABLE:
            try:
                send_progress_status_callback(message)
                log.debug(f"Status callback sent: {message}")
            except Exception as e:
                log.warning(f"Failed to send status callback: {e}")

    def load_model(self) -> Tuple[Any, Any]:
        """Load the model and processor with common logic."""
        log.info("Loading model and processor...")
        start_time = time.time()

        model_name = self.model_config.get("model_name")
        tokenizer_model_name = self.model_config.get("tokenizer_model_name", "qwen2.5-vl-7b")
        dtype = self.model_config.get("dtype", "bfloat16")
        tp_size = self.model_config.get("tp_size", 1)
        max_length = self.model_config.get("max_length", 128000)

        if should_enable_lora(self.config, self.enable_lora):
            base_model_path = get_base_model_path(self.config)
            model_name = merge_lora_model(model_name, base_model_path)
            log.info(f"LoRA merging enabled. Using merged model: {model_name}")

        model, processor = self._define_model(
            tokenizer_model_name, model_name, dtype, tp_size, max_length
        )

        elapsed_time = time.time() - start_time
        log.info(f"Model loaded in {elapsed_time:.2f} seconds")
        self._send_status_callback(f"Model loaded successfully in {elapsed_time:.1f} seconds")
        return model, processor

    def _define_model(
        self,
        tokenizer_model_name: str,
        model_name: str,
        dtype: str,
        tp_size: int,
        max_length: int = 128000,
    ) -> Tuple[Any, Any]:
        """Define and load the language model and processor."""
        if os.path.isabs(model_name) and os.path.exists(model_name):
            checkpoint_output_dir = model_name
        else:
            hf_cache_dir = os.environ.get(
                "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            )
            checkpoint_output_dir = os.path.join(hf_cache_dir, model_name)
            os.makedirs(checkpoint_output_dir, exist_ok=True)

        # Download checkpoint and tokenizer if needed
        self._send_status_callback("Downloading model checkpoint and tokenizer...")
        download_checkpoint(model_name, checkpoint_output_dir)
        download_tokenizer(tokenizer_model_name, checkpoint_output_dir)
        self._send_status_callback("Model files downloaded, initializing VLLM...")

        log.info("Using VLLM backend.")
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

        llm = LLM(
            model=checkpoint_output_dir,
            tokenizer=checkpoint_output_dir,
            trust_remote_code=True,
            dtype=dtype,
            limit_mm_per_prompt={"video": 10, "image": 10},
            tensor_parallel_size=tp_size,
            max_model_len=max_length,
        )

        self._send_status_callback("Loading tokenizer processor...")
        processor = AutoProcessor.from_pretrained(
            checkpoint_output_dir, max_length=max_length
        )

        return llm, processor

    def prepare_inputs_parallel(
        self,
        input_tasks: List[Dict[str, Any]],
        num_processes: int,
    ) -> List[Any]:
        """Prepare model inputs for tasks in parallel."""
        if not input_tasks:
            log.info("No input tasks to prepare model inputs for.")
            return []

        num_workers = min(num_processes, len(input_tasks))
        log.info(f"Preparing model inputs in parallel for {len(input_tasks)} tasks "
                f"using {num_workers} threads.")
        self._send_status_callback(f"Starting input preparation for {len(input_tasks)} tasks with {num_workers} workers...")

        worker_fn = partial(
            self._prepare_single_model_input,
            processor=self.processor,
            vision_config=self.vision_config,
        )

        processed_inputs_with_index = []
        completed_count = 0
        last_callback_time = time.time()
        callback_interval = 20  # Send status update every 20 seconds

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(worker_fn, task): i for i, task in enumerate(input_tasks)
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="Preparing model inputs",
            ):
                idx = future_to_idx[future]
                try:
                    model_input = future.result()
                    if model_input is not None:
                        processed_inputs_with_index.append((idx, model_input))
                except Exception as e:
                    log.error(f"Error preparing model input for task {idx}: {e}. Skipping task.")

                completed_count += 1
                # Send periodic status updates to prevent timeout
                current_time = time.time()
                if current_time - last_callback_time >= callback_interval:
                    progress_pct = (completed_count / len(input_tasks)) * 100
                    self._send_status_callback(
                        f"Preparing model inputs: {completed_count}/{len(input_tasks)} tasks completed ({progress_pct:.1f}%)"
                    )
                    last_callback_time = current_time

        processed_inputs_with_index.sort(key=lambda x: x[0])
        inputs = [inp for idx, inp in processed_inputs_with_index]

        if len(inputs) < len(input_tasks):
            log.warning(f"Successfully prepared inputs for {len(inputs)} out of {len(input_tasks)} tasks.")

        self._send_status_callback(f"Input preparation completed: {len(inputs)} tasks ready for inference")
        return inputs

    def _prepare_single_model_input(
        self,
        input_task: Dict[str, Any],
        processor: Any,
        vision_config: Dict[str, Any],
    ) -> Optional[Any]:
        """Prepare input data for a single model inference task."""
        prompt = input_task.get("prompt", [])
        media_paths = input_task.get("media_paths", [])
        media_mode = input_task.get("media_mode", "image")

        # Add video/image information to the user message content
        if (len(prompt) > 1
            and prompt[1]["role"] == "user"
            and isinstance(prompt[1]["content"], str)):

            content = []
            for media_path in media_paths:
                video_content = {
                    "type": media_mode,
                    media_mode: media_path,
                }
                # Add vision config parameters
                for k, v in vision_config.items():
                    video_content[k] = v
                content.append(video_content)

            content.append({
                "type": "text",
                "text": prompt[1]["content"]
            })
            prompt[1]["content"] = content

        # Apply chat template
        processed_text_prompt = processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

        # Process vision information
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            prompt, return_video_kwargs=True
        )

        if not video_inputs and not image_inputs:
            if media_paths:  # Only log error if media was expected
                log.error(f"No video or image inputs found for task: {input_task.get('id', 'unknown')}. Cannot prepare model input.")
                return None
            else:
                # Text-only task
                model_input = {"prompt": processed_text_prompt}
        elif video_inputs:
            model_input = {
                "prompt": processed_text_prompt,
                "multi_modal_data": {"video": video_inputs},
                "mm_processor_kwargs": video_kwargs,
            }
        else:
            model_input = {
                "prompt": processed_text_prompt,
                "multi_modal_data": {"image": image_inputs},
            }

        log.debug(f"Prepared model input for task: {input_task.get('id', 'unknown')}")
        return model_input

    def run_model_inference(
        self,
        inputs: List[Any],
        input_tasks: List[Dict[str, Any]],
        answer_type: str = "freeform",
    ) -> List[str]:
        """Run the model on inputs and return predictions."""
        # Get generation parameters
        max_tokens = self.gen_config.get("max_tokens", 1024)
        temperature = self.gen_config.get("temperature", 0)
        repetition_penalty = self.gen_config.get("repetition_penalty", 1.0)
        presence_penalty = self.gen_config.get("presence_penalty", 0.0)
        frequency_penalty = self.gen_config.get("frequency_penalty", 0.0)
        seed = self.eval_config.get("seed", 1)

        stop_token_id = self.processor.tokenizer.eos_token_id

        # Configure sampling parameters
        if answer_type == "letter":
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=10,
                stop_token_ids=[stop_token_id],
                top_k=1,
                seed=seed,
            )
        else:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                stop_token_ids=[stop_token_id],
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                seed=seed,
            )

        log.info(f"Generating outputs for {len(inputs)} tasks using VLLM...")
        self._send_status_callback(f"Starting model inference for {len(inputs)} tasks...")

        inference_start = time.time()

        # Process in batches to send status callbacks during long generation
        # This prevents timeout for large datasets where generation can take >15 minutes
        batch_size = self.eval_config.get("batch_size", 8)
        log.info(f"Using batch size: {batch_size}")
        all_outputs = []

        for batch_idx in range(0, len(inputs), batch_size):
            batch_end = min(batch_idx + batch_size, len(inputs))
            batch_inputs = inputs[batch_idx:batch_end]
            batch_num = (batch_idx // batch_size) + 1
            total_batches = (len(inputs) + batch_size - 1) // batch_size

            log.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_inputs)} requests)...")
            self._send_status_callback(
                f"Model inference: Processing batch {batch_num}/{total_batches} "
                f"(requests {batch_idx+1}-{batch_end}/{len(inputs)})"
            )

            batch_start_time = time.time()
            batch_outputs = self.model.generate(batch_inputs, sampling_params)
            batch_time = time.time() - batch_start_time

            all_outputs.extend(batch_outputs)

            completed = len(all_outputs)
            progress_pct = (completed / len(inputs)) * 100
            elapsed_total = time.time() - inference_start

            log.info(f"Batch {batch_num}/{total_batches} completed in {batch_time:.1f}s. "
                    f"Total progress: {completed}/{len(inputs)} ({progress_pct:.1f}%)")
            self._send_status_callback(
                f"Model inference progress: {completed}/{len(inputs)} requests completed "
                f"({progress_pct:.1f}%), elapsed time: {elapsed_total:.1f}s"
            )

        inference_time = time.time() - inference_start
        log.info(f"Finished VLLM generation. Received {len(all_outputs)} outputs in {inference_time:.1f}s.")
        self._send_status_callback(
            f"Model inference completed: {len(all_outputs)} outputs generated in {inference_time:.1f} seconds"
        )

        # Extract predictions
        self._send_status_callback("Extracting predictions from model outputs...")
        predictions = []
        for requestoutput in all_outputs:
            output_text = requestoutput.outputs[0].text
            predictions.append(output_text)

        self._send_status_callback(f"Successfully extracted {len(predictions)} predictions")
        return predictions

    def run_evaluation(
        self,
        results_dir: Path,
        skip_saved: bool = False,
        limit: int = -1,
        total_shard: int = 1,
        shard_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.

        Args:
            results_dir: Directory to save results
            skip_saved: Whether to skip already saved results
            limit: Limit number of tasks (for debugging)
            total_shard: Total number of shards
            shard_id: Current shard ID

        Returns:
            Dictionary containing evaluation metrics
        """
        start_time = time.time()

        # Load model
        self._send_status_callback("Initializing evaluation pipeline...")
        self.model, self.processor = self.load_model()

        # Get evaluation parameters
        answer_type = self.eval_config.get("answer_type", "freeform")
        num_processes = self.eval_config.get("num_processes", 40)

        # Create results directory structure
        save_folder = self.model_config.get("save_folder", None)
        if save_folder:
            results_output_dir = results_dir / save_folder
        else:
            model_name = self.model_config.get("model_name", "unknown_model")
            results_output_dir = results_dir / Path(model_name).name / answer_type

        results_output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Gather tasks
        log.info("Gathering evaluation tasks...")
        self._send_status_callback(f"Gathering evaluation tasks from dataset (shard {shard_id}/{total_shard})...")
        inputs, outputs = self.make_tasks(results_output_dir, total_shard, shard_id)
        log.info(f"Gathered {len(inputs)} tasks")
        self._send_status_callback(f"Gathered {len(inputs)} evaluation tasks")

        # Log progress to TAO
        log_tao_status(
            data={
                "evaluation_phase": "task_gathering",
                "total_tasks": len(inputs),
                "shard_id": shard_id,
                "total_shards": total_shard,
                "lora_enabled": self.enable_lora or self.model_config.get("enable_lora", False)
            },
            component_name=COMPONENT_NAME
        )

        # Step 2: Skip saved results if requested
        if skip_saved:
            log.info("Checking for saved results...")
            self._send_status_callback("Checking for previously saved results...")
            filtered_inputs = []
            filtered_outputs = []
            for i, o in zip(inputs, outputs):
                if not Path(o["output_path"]).exists():
                    filtered_inputs.append(i)
                    filtered_outputs.append(o)
            inputs, outputs = filtered_inputs, filtered_outputs
            log.info(f"Tasks remaining after skipping saved: {len(inputs)}")
            self._send_status_callback(f"Tasks remaining after skipping saved: {len(inputs)}")

        # Apply limit if specified
        if limit > 0 and len(inputs) > limit:
            inputs = inputs[:limit]
            outputs = outputs[:limit]
            log.info(f"Limited tasks to {len(inputs)} for debugging")

        if not inputs:
            log.info("No tasks to evaluate. Exiting.")
            self._send_status_callback("No tasks to evaluate - all results already exist")
            return {"overall": {"accuracy": 0.0, "total": 0, "correct": 0}}

        # Step 3: Prepare model inputs
        log.info("Preparing model inputs...")
        prepared = self.prepare_inputs_parallel(inputs, num_processes)
        log.info(f"Prepared {len(prepared)} model inputs")

        # Log progress to TAO
        log_tao_status(
            data={
                "evaluation_phase": "input_preparation",
                "prepared_inputs": len(prepared)
            },
            component_name=COMPONENT_NAME
        )

        # Step 4: Run model inference
        log.info("Running model inference...")
        inference_start = time.time()
        predictions = self.run_model_inference(prepared, inputs, answer_type)
        inference_time = time.time() - inference_start
        log.info(f"Model inference completed in {inference_time:.2f} seconds")

        # Log progress to TAO
        log_tao_status(
            data={
                "evaluation_phase": "inference",
                "inference_time_seconds": inference_time,
                "tasks_processed": len(inputs)
            },
            component_name=COMPONENT_NAME
        )

        # Step 5: Save results
        log.info("Saving results...")
        self._send_status_callback(f"Saving evaluation results for {len(predictions)} tasks...")
        self.save(outputs, predictions)
        self._send_status_callback("Results saved successfully")

        # Step 6: Compute metrics
        log.info("Computing evaluation metrics...")
        self._send_status_callback("Computing evaluation metrics...")
        metrics = self.compute_metrics(results_output_dir, outputs, predictions)

        total_time = time.time() - start_time
        log.info(f"Complete evaluation pipeline finished in {total_time:.2f} seconds")
        self._send_status_callback(f"Evaluation completed successfully in {total_time:.1f} seconds")

        return metrics

    @abstractmethod
    def make_tasks(self, results_dir: Path, total_shard: int, shard_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Create evaluation tasks from the dataset."""
        ...

    @abstractmethod
    def save(self, outputs: List[Dict[str, Any]], predictions: List[str]) -> None:
        """Save predictions to output files."""
        ...

    @abstractmethod
    def compute_metrics(self, results_dir: Path, outputs: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, Any]:
        """Compute evaluation metrics from predictions."""
        ...


