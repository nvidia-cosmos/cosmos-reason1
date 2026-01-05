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
"""Metropolis SGD metrics for Cosmos-RL spatial reasoning evaluation."""
from __future__ import annotations

import logging as log
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


class MetropolisSGDMetrics:
    """
    Metropolis SGD metrics for spatial reasoning tasks (warehouse/spatial-ai).
    
    This metric class computes metrics like:
    - Count accuracy (exact match)
    - Distance estimation accuracy (within 10% tolerance)
    - Left/right spatial reasoning accuracy
    - Multiple choice question (MCQ) accuracy
    - Weighted overall scores
    - Detailed error metrics for quantitative questions
    """

    EPSILON = 1e-8

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the MetropolisSGDMetrics.
        
        Args:
            weights: Custom weights for different metric types
        """
        # Default weights for different categories
        self._weights = weights or {
            "count": 0.25,
            "distance": 0.25,
            "left_right": 0.25,
            "mcq": 0.25,
        }

    def compute(
        self,
        predictions: List[str],
        references: List[str],
        categories: List[str],
        video_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute Metropolis SGD metrics using the actual evaluation logic.
        
        Args:
            predictions: List of model predictions
            references: List of ground truth references
            categories: List of question categories (count, distance, left_right, mcq)
            video_ids: Optional list of video/sample IDs
            
        Returns:
            Dictionary containing computed metrics
        """
        return self._compute_metrics(predictions, references, categories)

    def _compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        categories: List[str],
    ) -> Dict[str, Any]:
        """
        Compute metrics following the actual evaluation script logic.
        """
        if len(predictions) != len(references) or len(predictions) != len(categories):
            log.warning(
                f"Mismatch in lengths: predictions={len(predictions)}, "
                f"references={len(references)}, categories={len(categories)}"
            )
        
        # Initialize result dictionaries
        qualitative_dict = defaultdict(list)
        quantitative_success_dict = defaultdict(list)
        quantitative_error_dict = defaultdict(list)
        
        # Process each prediction
        for pred, ref, category in zip(predictions, references, categories):
            # Determine question type based on category
            if category in ['count', 'distance']:
                question_type = "quantitative"
            elif category in ['left_right', 'mcq']:
                question_type = "qualitative"
            else:
                log.warning(f"Unknown category: {category}")
                continue
            
            if question_type == "quantitative":
                try:
                    gt_value = float(ref)
                    pred_value = float(pred)
                    
                    # Compute success based on category
                    if category == "count":
                        gt_value = int(gt_value)
                        pred_value = int(pred_value)
                        success = gt_value == pred_value
                        error_rate = abs(gt_value - pred_value) / max(1, gt_value)
                    
                    elif category == "distance":
                        success = (pred_value <= (1.10 * gt_value)) and (
                            pred_value >= (0.90 * gt_value)
                        )
                        error_rate = (np.abs(pred_value - gt_value)) / (gt_value + self.EPSILON)
                    else:
                        success = False
                        error_rate = 1.0
                    
                    # Store results
                    quantitative_success_dict[category].append(int(success))
                    quantitative_error_dict[category].append(error_rate)
                    
                except (ValueError, TypeError) as e:
                    log.warning(f"Error parsing quantitative answer: pred={pred}, ref={ref}, error={e}")
                    quantitative_success_dict[category].append(0)
                    quantitative_error_dict[category].append(1.0)
                
            elif question_type == "qualitative":
                # Normalize answers based on category
                normalized_gt = self._normalize_qualitative_answer(ref, category)
                normalized_pred = self._normalize_qualitative_answer(pred, category)
                
                # For qualitative questions, we consider it correct if the normalized answers match
                success = int(normalized_gt == normalized_pred)
                qualitative_dict[category].append(success)
        
        # Calculate metrics
        result_dict = {}
        
        # Calculate qualitative metrics
        total_qualitative = 0
        correct_qualitative = 0
        for qual_cat in qualitative_dict.keys():
            correct_qualitative += np.sum(qualitative_dict[qual_cat])
            total_qualitative += len(qualitative_dict[qual_cat])
            accuracy = np.sum(qualitative_dict[qual_cat]) / (len(qualitative_dict[qual_cat]) + self.EPSILON) * 100
            result_dict[f"Qual_{qual_cat}_acc"] = accuracy
            result_dict[f"Qual_{qual_cat}_correct"] = int(np.sum(qualitative_dict[qual_cat]))
            result_dict[f"Qual_{qual_cat}_total"] = len(qualitative_dict[qual_cat])
        
        if total_qualitative > 0:
            result_dict["Qual_overall_acc"] = correct_qualitative / (total_qualitative + self.EPSILON) * 100
            result_dict["Qual_total_correct"] = int(correct_qualitative)
            result_dict["Qual_total_count"] = total_qualitative

        # Calculate quantitative metrics
        total_quantitative = 0
        correct_quantitative = 0
        
        for quant_cat in quantitative_success_dict.keys():
            correct_quantitative += np.sum(quantitative_success_dict[quant_cat])
            accuracy = (
                np.sum(quantitative_success_dict[quant_cat]) / (len(quantitative_success_dict[quant_cat]) + self.EPSILON) * 100
            )
            result_dict[f"Quan_{quant_cat}_acc"] = accuracy
            result_dict[f"Quan_{quant_cat}_correct"] = int(np.sum(quantitative_success_dict[quant_cat]))
            result_dict[f"Quan_{quant_cat}_total"] = len(quantitative_success_dict[quant_cat])
            total_quantitative += len(quantitative_success_dict[quant_cat])
            
            if quant_cat in quantitative_error_dict:
                error_rate = (
                    np.sum(quantitative_error_dict[quant_cat]) / (len(quantitative_error_dict[quant_cat]) + self.EPSILON) * 100
                )
                result_dict[f"Quan_{quant_cat}_err"] = error_rate

        if total_quantitative > 0:
            result_dict["Quan_overall_acc"] = (correct_quantitative / total_quantitative) * 100
            result_dict["Quan_total_correct"] = int(correct_quantitative)
            result_dict["Quan_total_count"] = total_quantitative

        # Calculate weighted scores
        weighted_scores = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        
        # Process quantitative categories
        for category in ['count', 'distance']:
            if category in quantitative_success_dict and len(quantitative_success_dict[category]) > 0:
                correct_count = sum(quantitative_success_dict[category])
                total_count = len(quantitative_success_dict[category])
                accuracy = (correct_count / total_count) * 100
                weighted_scores[category] = accuracy * self._weights[category]
                total_weighted_score += weighted_scores[category]
                total_weight += self._weights[category]
        
        # Process qualitative categories
        for category in ['left_right', 'mcq']:
            if category in qualitative_dict and len(qualitative_dict[category]) > 0:
                correct_count = sum(qualitative_dict[category])
                total_count = len(qualitative_dict[category])
                accuracy = (correct_count / total_count) * 100
                weighted_scores[category] = accuracy * self._weights[category]
                total_weighted_score += weighted_scores[category]
                total_weight += self._weights[category]
        
        # Calculate final weighted score
        if total_weight > 0:
            final_weighted_score = total_weighted_score / total_weight
            result_dict["Final_weighted_score"] = final_weighted_score
        
        # Overall statistics
        if (total_quantitative + total_qualitative) > 0:
            overall_acc = (correct_quantitative + correct_qualitative) / (total_quantitative + total_qualitative) * 100
            result_dict["Overall_acc"] = overall_acc
            result_dict["Total_count"] = total_quantitative + total_qualitative
            result_dict["Total_correct"] = int(correct_quantitative + correct_qualitative)
        
        return result_dict

    @staticmethod
    def _convert_number_word_to_digit(text: str) -> str:
        """Convert number words to digits."""
        if not isinstance(text, str):
            return str(text)
            
        number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10'
        }
        text = text.lower().strip()
        for word, digit in number_words.items():
            text = text.replace(word, digit)
        return text

    @classmethod
    def _extract_first_digit(cls, text: str) -> str:
        """Extract the first digit from text."""
        # First convert number words to digits
        text = cls._convert_number_word_to_digit(text)
        # Find first digit in the text
        match = re.search(r'\d+', text)
        if match:
            return match.group()
        return text

    @classmethod
    def _normalize_qualitative_answer(cls, answer: str, category: str) -> str:
        """Normalize qualitative answers based on category."""
        if not isinstance(answer, (str, int, float)):
            answer = str(answer)
        
        if category == 'left_right':
            # Handle common variations of left/right
            answer = answer.lower().strip()
            if answer in ['l', 'left', 'left side', 'left-hand', 'left hand']:
                return 'left'
            elif answer in ['r', 'right', 'right side', 'right-hand', 'right hand']:
                return 'right'
            return answer
        elif category == 'mcq':
            # Extract first digit after normalization
            return cls._extract_first_digit(answer)
        return str(answer).strip()
