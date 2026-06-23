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

"""
ITS (Intelligent Transportation Systems) Evaluator for Cosmos-RL.

This module combines the inference and scoring logic from the original ITS evaluation
into a single, integrated pipeline that works with the cosmos-rl CLI structure.
"""

import json
import logging as log
import os
import glob
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import attrs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from cosmos_rl.evaluation.base import BaseEvaluator

# Set up image processing
Image.MAX_IMAGE_PIXELS = 933120000
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Constants
COMPONENT_NAME = "Cosmos-RL ITS Evaluation"
DIRECTION_STRAIGHT = "going straight"
DIRECTION_LEFT = "turning left"
DIRECTION_RIGHT = "turning right"


@attrs.define(slots=False)
class ITSInputStructure:
    """Input structure for ITS evaluation tasks."""
    datasource: str
    media_id: str
    question: str
    question_idx: int
    media_paths: List[str]
    media_mode: str
    correct_answer: str
    prompt: Optional[Union[str, List[Dict[str, Any]]]] = None

    @classmethod
    def from_dict(cls, datasource: str, qa_pair: Dict[str, Any]) -> "ITSInputStructure":
        """Create ITSInputStructure from a question-answer pair dictionary."""
        return cls(
            datasource=datasource,
            media_id=qa_pair["media_id"],
            question=qa_pair['conversations'][1]["content"],
            correct_answer=qa_pair['conversations'][2]["content"],
            question_idx=qa_pair['id'],
            media_paths=qa_pair["media_paths"],
            media_mode=qa_pair["media_mode"],
            prompt=qa_pair['conversations'][:-1],
        )


@attrs.define(slots=False)
class ITSOutputStructure:
    """Output structure for ITS evaluation results."""
    datasource: str
    video_id: str
    correct_answer: str
    output_json_fname: str
    prompt: str = ""
    answer: str = ""
    reasoning: str = ""
    full_response: str = ""
    is_correct: bool = False


class ITSEvaluator(BaseEvaluator):
    """
    ITS evaluator that inherits from BaseEvaluator.

    Handles ITS-specific dataset loading, task creation, and directionality metrics.
    """

    def __init__(self, config: Dict[str, Any], enable_lora: bool = False):
        """
        Initialize the ITS evaluator with configuration.

        Args:
            config: Evaluation configuration dictionary
            enable_lora: Whether to enable LoRA model merging
        """
        super().__init__(config, enable_lora=enable_lora)
        self.datasets = {"eval_set_name": config["dataset"]}

    def make_tasks(self, results_dir: Path, total_shard: int, shard_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Gather all evaluation tasks from datasets."""
        input_tasks = []
        output_results = []

        qa_pairs = []
        answer_type = self.eval_config.get("answer_type", "freeform")
        
        for datasource_name, datasource_config in self.datasets.items():
            log.info(f"Gathering tasks from dataset: {datasource_name}")

            media_dir = datasource_config.get("media_dir", None)
            annotation_path = datasource_config.get("annotation_path")
            system_prompt = datasource_config.get("system_prompt", "")

            if answer_type == "reasoning" and '<think>' not in system_prompt:
                system_prompt += ("Answer the question with provided options in the following format: "
                                "\\n<think>\\nyour reasoning\\n</think> <answer>\\nyour answer\\n</answer>.")

            log.info(f"System prompt: {system_prompt}")

            # Check if media directory exists
            if media_dir and not os.path.exists(media_dir):
                log.error(f"Media path does not exist: {media_dir}")
                continue

            with open(annotation_path, 'r') as f:
                annotations = json.load(f)

                for item in annotations:
                    # Clean question text
                    question = re.sub(r"(\\n)?</?(image|video)>(\\n)?", "",
                                    item['conversations'][0]['value']).strip()

                    conversation = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": item['conversations'][1]['value']}
                    ]

                    # Handle media paths
                    images = item.get("image", None) or item.get("images", None)
                    videos = item.get("video", None)

                    if images:
                        if isinstance(images, str):
                            images = [images]
                        relative_media_paths = images
                        media_mode = 'image'
                    elif videos:
                        if isinstance(videos, str):
                            videos = [videos]
                        relative_media_paths = videos
                        media_mode = 'video'
                    else:
                        log.error(f"No media paths found for item: {item}")
                        continue

                    if media_dir:
                        media_paths = [os.path.join(media_dir, path) for path in relative_media_paths]
                    else:
                        media_paths = relative_media_paths

                    qa_pairs.append({
                        "datasource": datasource_name,
                        "media_id": relative_media_paths[0],
                        "id": item['id'],
                        "media_paths": media_paths,
                        "media_mode": media_mode,
                        "conversations": conversation,
                        "correct_answer": item['conversations'][1]['value']
                    })

        # Shard the tasks
        shard_qa_pairs = qa_pairs[shard_id::total_shard]
        log.info(f"Sharding {len(qa_pairs)} tasks into {total_shard} shards, "
                f"shard {shard_id} has {len(shard_qa_pairs)} tasks.")

        for qa_pair in shard_qa_pairs:
            output_json_fname = results_dir / qa_pair["datasource"] / f"{qa_pair['media_id']}.json"
            output_json_fname.parent.mkdir(parents=True, exist_ok=True)

            # Create task in format expected by base class
            input_task = {
                "id": qa_pair["id"],
                "datasource": qa_pair["datasource"],
                "media_id": qa_pair["media_id"],
                "prompt": qa_pair["conversations"][:-1],  # Exclude the answer
                "media_paths": qa_pair["media_paths"],
                "media_mode": qa_pair["media_mode"],
                "correct_answer": qa_pair["correct_answer"]
            }
            
            output_result = {
                "id": qa_pair["id"],
                "output_path": str(output_json_fname),
                "datasource": qa_pair["datasource"],
                "video_id": qa_pair['media_id'],
                "correct_answer": qa_pair["correct_answer"]
            }

            input_tasks.append(input_task)
            output_results.append(output_result)

        return input_tasks, output_results

    def save(self, outputs: List[Dict[str, Any]], predictions: List[str]) -> None:
        """Save ITS evaluation results to JSON files."""
        answer_type = self.eval_config.get("answer_type", "freeform")
        
        for output, prediction in zip(outputs, predictions):
            # Parse based on answer type
            if answer_type == "letter":
                answer, reasoning = self._parse_letter_response(prediction)
            elif answer_type == "reasoning":
                answer, reasoning = self._parse_reasoning_response(prediction)
            else:
                answer = prediction
                reasoning = ""

            # Check correctness
            is_correct = (answer.lower() == output["correct_answer"].lower())

            result_data = {
                "datasource": output.get("datasource", "unknown"),
                "video_id": output.get("video_id", output.get("id", "unknown")),
                "correct_answer": output["correct_answer"],
                "answer": answer,
                "reasoning": reasoning,
                "full_response": prediction,
                "is_correct": is_correct,
            }

            os.makedirs(os.path.dirname(output["output_path"]), exist_ok=True)
            with open(output["output_path"], 'w') as f:
                json.dump([result_data], f, indent=2)

    def _parse_letter_response(self, response: str) -> Tuple[str, str]:
        """Parse letter-format response."""
        return response.strip()[:1], ""

    def _parse_reasoning_response(self, response: str) -> Tuple[str, str]:
        """Parse reasoning-format response."""
        # Extract answer and reasoning from <think> and <answer> tags
        reasoning_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)

        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        answer = answer_match.group(1).strip() if answer_match else response.strip()

        return answer, reasoning

    def compute_metrics(self, results_dir: Path, outputs: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, Any]:
        """Compute ITS directionality metrics."""
        return self._evaluate_directionality(results_dir)

    def _evaluate_directionality(self, result_path: Path) -> Dict[str, Any]:
        """
        Evaluate directionality metrics from results.

        This is the scoring logic from the original score.py.
        """
        log.info("Computing directionality metrics...")
        self._send_status_callback("Computing directionality metrics from evaluation results...")

        results = glob.glob(str(result_path / "eval_set_name" / "**" / "*.json"), recursive=True)
        log.info(f"Found {len(results)} result files.")
        self._send_status_callback(f"Processing {len(results)} result files for metrics computation...")

        correct_count = 0
        total_count = 0
        result_dict = {
            DIRECTION_STRAIGHT: {"correct": 0, "total": 0},
            DIRECTION_LEFT: {"correct": 0, "total": 0},
            DIRECTION_RIGHT: {"correct": 0, "total": 0}
        }

        word_classes = [DIRECTION_STRAIGHT, DIRECTION_LEFT, DIRECTION_RIGHT]
        confusion_matrix = np.zeros((len(word_classes), len(word_classes)), dtype=int)
        word_to_idx = {w: i for i, w in enumerate(word_classes)}

        for result_file in results:
            try:
                with open(result_file, "r") as f:
                    data = json.load(f)

                for item in data:
                    total_count += 1
                    gt = item["correct_answer"].lower()
                    response = item["answer"].lower()

                    # Determine ground truth type
                    if 'straight' in gt:
                        gt_type = DIRECTION_STRAIGHT
                    elif 'right' in gt:
                        gt_type = DIRECTION_RIGHT
                    elif 'left' in gt:
                        gt_type = DIRECTION_LEFT
                    else:
                        continue

                    result_dict[gt_type]["total"] += 1

                    # Determine response type
                    if 'straight' in response:
                        response_type = DIRECTION_STRAIGHT
                    elif 'right' in response:
                        response_type = DIRECTION_RIGHT
                    elif 'left' in response:
                        response_type = DIRECTION_LEFT
                    else:
                        response_type = "unknown"
                        # For confusion matrix, treat unknown as a miss
                        continue

                    if response_type == gt_type:
                        correct_count += 1
                        result_dict[gt_type]["correct"] += 1

                    # Update confusion matrix (only for known responses)
                    if response_type in word_to_idx:
                        confusion_matrix[word_to_idx[gt_type], word_to_idx[response_type]] += 1

            except Exception as e:
                log.error(f"Error processing result file {result_file}: {e}")

        # Calculate accuracies
        for k, v in result_dict.items():
            if v["total"] > 0:
                accuracy = v["correct"] / v["total"]
                result_dict[k]["accuracy"] = accuracy
            else:
                result_dict[k]["accuracy"] = 0.0

        overall_accuracy = correct_count / total_count if total_count > 0 else 0.0
        result_dict["overall"] = {
            "correct": correct_count,
            "total": total_count,
            "accuracy": overall_accuracy
        }

        # Save results
        self._send_status_callback("Saving directionality metrics to JSON...")
        results_file = result_path / "directionality_score.json"
        with open(results_file, "w") as f:
            json.dump(result_dict, f, indent=2)

        # Create and save confusion matrix
        self._send_status_callback("Generating confusion matrix visualization...")
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                   xticklabels=word_classes, yticklabels=word_classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(result_path / "directionality_confusion_matrix.png")
        plt.close()

        log.info("Directionality evaluation completed:")
        log.info(f"  Overall accuracy: {overall_accuracy:.4f} ({correct_count}/{total_count})")
        for category, metrics in result_dict.items():
            if category != "overall":
                acc = metrics.get("accuracy", 0.0)
                correct = metrics.get("correct", 0)
                total = metrics.get("total", 0)
                log.info(f"  {category}: {acc:.4f} ({correct}/{total})")

        return result_dict


def main():
    """
    Main entry point for cosmos-rl-evaluate command.

    This function provides a command-line interface for the ITS evaluator
    that can be called directly or through the pyproject.toml script entry point.
    """
    from cosmos_rl.evaluation.evaluate import main as evaluate_main
    evaluate_main()
