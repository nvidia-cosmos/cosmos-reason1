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
"""Text metrics for Cosmos-RL."""
from __future__ import annotations

from typing import Dict, List, Optional

import evaluate

class TextMetrics:
    """Text metrics for Cosmos-RL."""
    def __init__(self, metrics: List[str] | None = None, bertscore_model: Optional[str] = None, bertscore_lang: Optional[str] = None) -> None:
        """Initialize the TextMetrics."""
        self.metric_names = metrics or ["bleu", "rouge"]
        self._bleu = evaluate.load("bleu") if "bleu" in self.metric_names else None
        self._rouge = evaluate.load("rouge") if "rouge" in self.metric_names else None
        self._bertscore = evaluate.load("bertscore") if "bertscore" in self.metric_names else None
        self._bertscore_model = bertscore_model
        self._bertscore_lang = bertscore_lang
        # weights keys expected: "BLEU", "ROUGE_L", "BERTScore_F1"
        self._weights = {"BLEU": 0.3, "ROUGE_L": 0.3, "BERTScore_F1": 0.4}

    def compute(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Compute the metrics for the TextMetrics."""
        out: Dict[str, float] = {}
        if self._bleu is not None:
            bleu_res = self._bleu.compute(predictions=predictions, references=references)
            out["BLEU"] = float(bleu_res.get("bleu", 0.0))
        if self._rouge is not None:
            single_refs = [refs[0] if isinstance(refs, list) and refs else "" for refs in references]
            rouge_res = self._rouge.compute(predictions=predictions, references=single_refs)
            for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
                if k in rouge_res:
                    out[k.upper()] = float(rouge_res[k])
        if self._bertscore is not None:
            single_refs = [refs[0] if isinstance(refs, list) and refs else "" for refs in references]
            kwargs: Dict[str, Optional[str]] = {}
            if self._bertscore_model:
                kwargs["model_type"] = self._bertscore_model
            if self._bertscore_lang:
                kwargs["lang"] = self._bertscore_lang
            bs = self._bertscore.compute(predictions=predictions, references=single_refs, **kwargs)
            # bs contains precision/recall/f1 lists; average them
            def _avg(arr: List[float]) -> float:
                return float(sum(arr) / max(1, len(arr)))
            out["BERTScore_P"] = _avg(bs.get("precision", []))
            out["BERTScore_R"] = _avg(bs.get("recall", []))
            out["BERTScore_F1"] = _avg(bs.get("f1", []))

        # Optional weighted composite score
        if self._weights:
            weighted = 0.0
            total_w = 0.0
            for key, w in self._weights.items():
                if key in out and isinstance(w, (int, float)):
                    weighted += float(w) * float(out[key])
                    total_w += float(w)
            if total_w > 0:
                out["WEIGHTED_SCORE"] = weighted / total_w
            else:
                out["WEIGHTED_SCORE"] = 0.0
        return out


