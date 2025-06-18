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

from abc import ABC, abstractmethod
from typing import Any, List, Dict
from transformers import AutoTokenizer
from cosmos_reason1.policy.config import Config


class DataPacker(ABC):
    """
    This is where dataset item is transformed into the format required by the rollout engine (e.g. vllm)
    for example:
        - `str` is needed for language model
        - {
            "prompt": prompt,
            "multi_modal_data": {"video": ...},
          } for multi-modal model
    """

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        assert config is not None, "config should be set"
        assert tokenizer is not None, "tokenizer should be set"
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def get_rollout_input(self, item: Any) -> Any:
        """
        Convert sample to data format required by the rollout engine (e.g. vllm)
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def rollout_collate_fn(self, items: List[Any]) -> List[Any]:
        """
        Collate the rollout inputs into a mini-batch for rollout engine
        """
        return items

    @abstractmethod
    def get_policy_input(
        self,
        sample: Any,
        rollout_output: str,
        n_ignore_prefix_tokens: int = 0,
    ) -> Any:
        """
        Stage for processing samples & rollout output before collating them into a mini-batch
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def policy_compute_max_len(self, processed_samples: List[Any]) -> int:
        """
        Compute the maximum sequence length of the mini-batch
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def policy_collate_fn(
        self,
        processed_samples: List[Any],
        computed_max_len: int,
    ) -> Dict[str, Any]:
        """
        Collate the processed samples into a mini-batch,
        the output will be fed into the policy model for training
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def sft_process_sample(self, sample: Any) -> Any:
        """
        Process the sample into the format required by the SFT model
        """
        raise NotImplementedError(
            "This method should be implemented by the subclass for SFT training"
        )

    def sft_compute_max_len(self, processed_samples: List[Any]) -> int:
        """
        Compute the maximum sequence length of the processed samples
        """
        raise NotImplementedError(
            "This method should be implemented by the subclass for SFT training"
        )

    def sft_collate_fn(
        self,
        sub_batch: List[Dict[str, Any]],
        computed_max_len: int,
        pad_token_id: int,
        ignore_label_id: int,
    ) -> Dict[str, Any]:
        """
        Collate the mini-batch into the format required by the policy model
        """
        raise NotImplementedError(
            "This method should be implemented by the subclass for SFT training"
        )
