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
"""Metropolis SGD evaluator for spatial reasoning tasks."""
from __future__ import annotations

import json
import logging as log
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from cosmos_rl.evaluation.base import BaseEvaluator
from cosmos_rl.evaluation.metrics.metropolis_sgd_metrics import MetropolisSGDMetrics

COMPONENT_NAME = "Cosmos-RL Metropolis SGD Evaluation"


class MetropolisSGDEvaluator(BaseEvaluator):
    """
    Metropolis SGD evaluator for spatial reasoning tasks.

    This evaluator is designed for warehouse/spatial-ai tasks that require:
    - Object counting
    - Distance estimation
    - Spatial reasoning (left/right)
    - Multiple choice questions

    It's based on the evaluation approach from the spatial-ai-warehouse toolbox
    and integrates with the cosmos_rl evaluation pipeline.
    """

    def __init__(self, config: Dict[str, Any], enable_lora: bool = False) -> None:
        """
        Initialize the MetropolisSGDEvaluator.

        Args:
            config: Configuration dictionary
            enable_lora: Whether to enable LoRA
        """
        super().__init__(config, enable_lora=enable_lora)

        # Get metrics configuration
        metrics_cfg = config.get("metrics", {})

        # Initialize MetropolisSGDMetrics
        self.metrics = MetropolisSGDMetrics(
            weights=metrics_cfg.get("weights"),
        )

        # Answer type configuration
        self.answer_type = self.eval_config.get("answer_type", "naive")

    def make_tasks(
        self, results_dir: Path, total_shard: int, shard_id: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create evaluation tasks from the dataset.

        Expected dataset format (similar to toolbox/evaluate.py):
        [
            {
                "id": "annotation_id",
                "conversations": [
                    {"role": "user", "value": "question with <image> or <video> tags"},
                    {"role": "assistant", "value": "ground_truth_answer"}
                ],
                "image": "path/to/image.jpg" or "images": ["path1.jpg", "path2.jpg"],
                "video": "path/to/video.mp4"
            }
        ]

        Returns:
            Tuple of (input_tasks, output_specs)
        """
        annotation_path = self.dataset_cfg.get("annotation_path")
        media_dir = self.dataset_cfg.get("media_dir", "")
        system_prompt = self.dataset_cfg.get("system_prompt", "Answer the questions.")

        log.info(f"Loading annotations from: {annotation_path}")
        with open(annotation_path, "r") as f:
            annotations = json.load(f)

        log.info(f"Loaded {len(annotations)} annotations")

        def resolve_media_paths(record: Dict[str, Any]) -> Tuple[List[str], str]:
            """Resolve media paths from annotation record."""
            images = record.get("image") or record.get("images")
            videos = record.get("video")

            if images:
                if isinstance(images, str):
                    images = [images]
                media_paths = images
                media_mode = "image"
            elif videos:
                if isinstance(videos, str):
                    videos = [videos]
                media_paths = videos
                media_mode = "video"
            else:
                media_paths = []
                media_mode = "image"

            # Join with media_dir if provided
            if media_dir:
                media_paths = [os.path.join(media_dir, p) for p in media_paths]

            return media_paths, media_mode

        # Process annotations into tasks
        qa_pairs: List[Dict[str, Any]] = []
        for item in annotations:
            sample_id = item.get("id", f"sample_{len(qa_pairs)}")

            # Extract question and answer from conversations
            if "conversations" in item and len(item["conversations"]) >= 2:
                # Handle both "from" (cosmos-tao format) and "role" (standard format)
                user_conv = item["conversations"][0]
                assistant_conv = item["conversations"][1]

                user_key = "value" if "value" in user_conv else "content"

                # Remove image/video tags from question
                user_prompt = user_conv[user_key]
                user_prompt = re.sub(r"(\n)?</?(image|video)>(\n)?", "", user_prompt).strip()

                # Get ground truth answer (prefer normalized_answer if available)
                ground_truth = item.get("normalized_answer", assistant_conv.get("value", assistant_conv.get("content", "")))

                # Get category
                category = item.get("category", "general")

                # Resolve media paths
                media_paths, media_mode = resolve_media_paths(item)

                qa_pairs.append({
                    "id": sample_id,
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "ground_truth": ground_truth,
                    "category": category,
                    "media_paths": media_paths,
                    "media_mode": media_mode,
                })
            else:
                log.warning(f"Skipping annotation {sample_id}: missing conversations")

        log.info(f"Created {len(qa_pairs)} tasks from annotations")

        # Shard the tasks
        shard = qa_pairs[shard_id::max(1, total_shard)]
        log.info(f"Shard {shard_id}/{total_shard}: {len(shard)} tasks")

        # Create output specifications
        tasks: List[Dict[str, Any]] = []
        outs: List[Dict[str, Any]] = []

        for rec in shard:
            # Create output path
            output_subdir = self.answer_type  # "naive" or "reasoning"
            out_path = results_dir / output_subdir / f"{rec['id']}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            tasks.append(rec)
            outs.append({
                "id": rec["id"],
                "output_path": str(out_path),
                "ground_truth": rec["ground_truth"],
                "category": rec.get("category", "general"),
            })

        return tasks, outs

    def save(self, outputs: List[Dict[str, Any]], predictions: List[str]) -> None:
        """
        Save predictions to output files.

        Output format matches toolbox/evaluate.py structure:
        [
            {
                "datasource": "warehouse",
                "video_id": "annotation_id",
                "prompt": null,
                "correct_answer": "ground_truth",
                "reasoning": "",
                "answer": "extracted_answer",
                "full_response": "complete_model_response",
                "is_correct": true/false
            }
        ]
        """
        for out, pred in zip(outputs, predictions):
            # Get category for proper normalization
            category = out.get("category", "general")
            
            # Extract answer from prediction based on answer type
            if self.answer_type == "reasoning":
                # For reasoning mode, try to extract answer from prediction
                extracted_answer = self._extract_answer(pred)
                reasoning = pred
            else:
                # For naive mode, the full response is the answer
                extracted_answer = self._extract_answer(pred)
                reasoning = ""
            
            # Normalize answers for comparison based on category
            if category in ['left_right', 'mcq']:
                pred_norm = MetropolisSGDMetrics._normalize_qualitative_answer(extracted_answer, category)
                gt_norm = MetropolisSGDMetrics._normalize_qualitative_answer(out["ground_truth"], category)
            else:
                # For quantitative (count, distance), just strip whitespace
                pred_norm = str(extracted_answer).strip()
                gt_norm = str(out["ground_truth"]).strip()
            
            is_correct = pred_norm == gt_norm

            # Create result entry
            result_entry = {
                "datasource": "warehouse",
                "video_id": out["id"],
                "prompt": None,
                "correct_answer": out["ground_truth"],
                "reasoning": reasoning,
                "answer": extracted_answer,
                "full_response": pred,
                "is_correct": is_correct,
            }

            # Save to file (as list with single entry, matching toolbox format)
            out_path = Path(out["output_path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump([result_entry], f, indent=4, ensure_ascii=False)

        log.info(f"Saved {len(outputs)} results")

    def _extract_answer(self, response: str) -> str:
        """
        Extract answer from model response.

        This method attempts to extract the meaningful answer from the full response:
        - For MCQ: Extract single letter (A-D)
        - For counting/distance: Extract number
        - For left/right: Extract "left" or "right"
        - Fallback: Return first word
        """
        if not response:
            return ""

        response = response.strip()

        # Try to extract single letter for MCQ
        letter_match = re.search(r"\b([A-D])\b", response, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()

        # Try to extract number for counting/distance
        number_match = re.search(r"\b(\d+\.?\d*)\b", response)
        if number_match:
            return number_match.group(1)

        # Try to extract left/right
        if re.search(r"\bleft\b", response, re.IGNORECASE):
            return "left"
        if re.search(r"\bright\b", response, re.IGNORECASE):
            return "right"

        # Remove common prefixes
        cleaned = re.sub(
            r"^(the\s+)?answer\s*(is)?\s*:?\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )

        # Extract first word
        first_word_match = re.search(r"\b(\w+)\b", cleaned)
        if first_word_match:
            return first_word_match.group(1)

        return response

    def compute_metrics(
        self,
        results_dir: Path,
        outputs: List[Dict[str, Any]],
        predictions: List[str],
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics using MetropolisSGDMetrics.

        Args:
            results_dir: Directory containing results
            outputs: Output specifications with ground truth and categories
            predictions: Model predictions

        Returns:
            Dictionary containing computed metrics
        """
        # Extract ground truth references and categories
        references = [out["ground_truth"] for out in outputs]
        categories = [out.get("category", "general") for out in outputs]
        video_ids = [out["id"] for out in outputs]

        # Compute metrics
        log.info("Computing Metropolis SGD metrics...")
        self._send_status_callback("Computing Metropolis SGD spatial reasoning metrics...")
        metrics = self.metrics.compute(
            predictions=predictions,
            references=references,
            categories=categories,
            video_ids=video_ids,
        )
        self._send_status_callback("Metropolis SGD metrics computed successfully")

        # Add overall statistics for compatibility with base evaluator
        overall_stats = {
            "accuracy": metrics.get("Overall_acc", 0.0) / 100.0,  # Convert to 0-1 scale
            "total": metrics.get("Total_count", len(predictions)),
            "correct": metrics.get("Total_correct", 0),
        }

        # Compile full results with proper structure for base evaluator
        results = {
            "overall": overall_stats,
        }
        
        # Add per-category results in expected format
        # For quantitative categories
        for cat in ['count', 'distance']:
            if f"Quan_{cat}_acc" in metrics:
                results[cat] = {
                    "accuracy": metrics[f"Quan_{cat}_acc"] / 100.0,  # Convert to 0-1 scale
                    "total": metrics.get(f"Quan_{cat}_total", 0),
                    "correct": metrics.get(f"Quan_{cat}_correct", 0),
                }
        
        # For qualitative categories
        for cat in ['left_right', 'mcq']:
            if f"Qual_{cat}_acc" in metrics:
                results[cat] = {
                    "accuracy": metrics[f"Qual_{cat}_acc"] / 100.0,  # Convert to 0-1 scale
                    "total": metrics.get(f"Qual_{cat}_total", 0),
                    "correct": metrics.get(f"Qual_{cat}_correct", 0),
                }
        
        # Add summary categories
        if "Quan_overall_acc" in metrics:
            results["quantitative"] = {
                "accuracy": metrics["Quan_overall_acc"] / 100.0,
                "total": metrics.get("Quan_total_count", 0),
                "correct": metrics.get("Quan_total_correct", 0),
            }
        
        if "Qual_overall_acc" in metrics:
            results["qualitative"] = {
                "accuracy": metrics["Qual_overall_acc"] / 100.0,
                "total": metrics.get("Qual_total_count", 0),
                "correct": metrics.get("Qual_total_correct", 0),
            }
        
        # Add weighted score if available (store in metrics dict, not as a category)
        if "Final_weighted_score" in metrics:
            results["overall"]["weighted_score"] = metrics["Final_weighted_score"] / 100.0
        
        # Store full metrics for reference
        results["metrics"] = metrics
        
        log.info("Metrics computation complete")
        log.info(f"Overall accuracy: {overall_stats['accuracy']:.4f}")
        log.info(f"Weighted score: {metrics.get('Final_weighted_score', 0.0):.2f}%")
        
        return results

