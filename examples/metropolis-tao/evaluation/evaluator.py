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
"""General evaluator for text-generation tasks (captioning, freeform QA)."""
from __future__ import annotations

import json
import logging as log
import os
import re
from pathlib import Path
import unicodedata
from typing import Any, Dict, List, Tuple

from cosmos_rl.evaluation.base import BaseEvaluator
from cosmos_rl.evaluation.metrics.text_metrics import TextMetrics

COMPONENT_NAME = "Cosmos-RL Evaluation"
class Evaluator(BaseEvaluator):
    """
    General evaluator for text-generation tasks (captioning, freeform QA).
    - Uses the common BaseEvaluator pipeline
    - Computes text metrics via Hugging Face 'evaluate' (BLEU/ROUGE)
    """

    def __init__(self, config: Dict[str, Any], enable_lora: bool = False) -> None:
        """
        Initialize the Evaluator.
        """
        super().__init__(config, enable_lora=enable_lora)
        metrics_cfg = config.get("metrics", {})

        # Get metric names as list
        metric_names = metrics_cfg.get("names", ["bleu", "rouge"])
        # Handle legacy comma-separated strings for backward compatibility
        if isinstance(metric_names, str):
            metric_names = [name.strip() for name in metric_names.split(",")]

        self.metrics = TextMetrics(
            metrics=metric_names,
            bertscore_model=metrics_cfg.get("bertscore_model", "microsoft/deberta-xlarge-mnli"),
            bertscore_lang=metrics_cfg.get("bertscore_lang", "en"),
        )

    def make_tasks(self, results_dir: Path, total_shard: int, shard_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Make tasks for the Evaluator.
        """
        annotation_path = self.dataset_cfg.get("annotation_path")
        media_dir = self.dataset_cfg.get("media_dir", None)
        system_prompt = self.dataset_cfg.get("system_prompt", "")

        with open(annotation_path, "r") as f:
            annotations = json.load(f)

        def resolve_media_paths(record: Dict[str, Any]) -> Tuple[List[str], str]:
            images = record.get("image", None) or record.get("images", None)
            videos = record.get("video", None)
            if images:
                if isinstance(images, str):
                    images = [images]
                rel = images
                mode = "image"
            elif videos:
                if isinstance(videos, str):
                    videos = [videos]
                rel = videos
                mode = "video"
            else:
                rel = []
                mode = "image"
            if media_dir:
                paths = [os.path.join(media_dir, p) for p in rel]
            else:
                paths = rel
            return paths, mode

        qa_pairs: List[Dict[str, Any]] = []
        for item in annotations:
            if "conversations" in item:
                question = re.sub(r"(\n)?</?(image|video)>(\n)?", "", item["conversations"][0]["value"]).strip()
                answer = item["conversations"][1]["value"]
                refs = [answer]
                media_paths, media_mode = resolve_media_paths(item)
                qa_pairs.append(
                    {
                        "id": item["id"],
                        "prompt": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                        "references": refs,
                        "media_paths": media_paths,
                        "media_mode": media_mode,
                    }
                )
            else:
                rid = item.get("id", item.get("media_id", ""))
                prompt = item.get("prompt", "")
                refs = item.get("references", [])
                media_paths, media_mode = resolve_media_paths(item)
                qa_pairs.append(
                    {
                        "id": rid,
                        "prompt": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "references": refs if isinstance(refs, list) else [str(refs)],
                        "media_paths": media_paths,
                        "media_mode": media_mode,
                    }
                )

        shard = qa_pairs[shard_id::max(1, total_shard)]
        tasks: List[Dict[str, Any]] = []
        outs: List[Dict[str, Any]] = []
        for rec in shard:
            out_path = results_dir / "general" / f"{rec['id']}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append(rec)
            outs.append({"id": rec["id"], "output_path": str(out_path), "references": rec["references"]})
        return tasks, outs

    def save(self, outputs: List[Dict[str, Any]], predictions: List[str]) -> None:
        """
        Save the outputs and predictions.
        """
        for out, pred in zip(outputs, predictions):
            data = {"id": out["id"], "prediction": pred, "references": out["references"]}
            out_path = out["output_path"]
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump([data], f, indent=2)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for exact-match comparison:
        - Lowercase
        - Remove articles (a, an, the)
        - Remove punctuation and symbols (Unicode-aware)
        - Collapse multiple spaces
        """
        if not isinstance(text, str):
            text = str(text)
        s = text.lower()
        # Remove punctuation and symbols using Unicode categories (P*, S*)
        s = "".join(ch if not unicodedata.category(ch).startswith(("P", "S")) else " " for ch in s)
        # Remove English articles
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def _is_exact_match(cls, prediction: str, references: List[str]) -> bool:
        """
        Check if the prediction is an exact match for any of the references.
        """
        pred_norm = cls._normalize_text(prediction)
        for ref in references:
            if cls._normalize_text(ref) == pred_norm:
                return True
        return False

    def compute_metrics(self, results_dir: Path, outputs: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, Any]:
        """
        Compute the metrics for the Evaluator.
        """
        self._send_status_callback("Computing text similarity metrics (BLEU/ROUGE/BERTScore)...")
        references: List[List[str]] = [o["references"] for o in outputs]

        # Compute similarity metrics (BLEU/ROUGE/BERTScore)
        metrics = self.metrics.compute(predictions=predictions, references=references)
        self._send_status_callback("Text similarity metrics computed")

        # Compute normalized exact-match accuracy
        correct = 0
        for pred, refs in zip(predictions, references):
            if self._is_exact_match(pred, refs):
                correct += 1

        total = len(predictions)
        accuracy = float(correct) / max(1, total)

        # Optional soft accuracy via token-overlap F1 threshold
        def _tokens(s: str) -> List[str]:
            return self._normalize_text(s).split()

        def _f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
            if not pred_tokens and not ref_tokens:
                return 1.0
            if not pred_tokens or not ref_tokens:
                return 0.0
            common = {}
            for t in pred_tokens:
                common[t] = common.get(t, 0) + 1
            overlap = 0
            for t in ref_tokens:
                if common.get(t, 0) > 0:
                    overlap += 1
                    common[t] -= 1
            if overlap == 0:
                return 0.0
            precision = overlap / len(pred_tokens)
            recall = overlap / len(ref_tokens)
            return 2 * precision * recall / (precision + recall)

        soft_cfg = self.eval_config.get("soft_accuracy", {}) if isinstance(self.eval_config, dict) else {}
        soft_enabled = bool(soft_cfg.get("enabled", True))
        soft_threshold = float(soft_cfg.get("f1_threshold", 0.8))

        soft_correct = 0
        if soft_enabled:
            for pred, refs in zip(predictions, references):
                ptoks = _tokens(pred)
                max_f1 = 0.0
                for ref in refs:
                    rtoks = _tokens(ref)
                    max_f1 = max(max_f1, _f1(ptoks, rtoks))
                if max_f1 >= soft_threshold:
                    soft_correct += 1
        soft_accuracy = float(soft_correct) / max(1, total) if soft_enabled else 0.0

        result: Dict[str, Any] = {
            "overall": {
                "accuracy": accuracy,
                "total": total,
                "correct": correct,
                "soft_accuracy": soft_accuracy if soft_enabled else 0.0,
                "soft_correct": soft_correct if soft_enabled else 0,
                "soft_threshold": soft_threshold if soft_enabled else None,
            }
        }
        result.update(metrics)
        return result
