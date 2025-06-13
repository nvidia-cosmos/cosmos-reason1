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

from cosmos_reason1.dispatcher.algo.base import RuleBasedAlgo, _register_rule_based_algo
from typing import Callable, List
import numpy as np
from cosmos_reason1.utils.constant import Algo


class GRPO(RuleBasedAlgo):
    def __init__(
        self, reward_fn: Callable, unbiased: bool = False, eps: float = 1e-5, **kwargs
    ):
        super().__init__(**kwargs)
        self.reward_fn = reward_fn
        self.unbiased = unbiased
        self.eps = eps

    def compute_reward(self, to_be_evaluated: str, reference: str) -> float:
        return self.reward_fn.compute_reward(to_be_evaluated, reference)

    def compute_advantage(self, rewards: List[float]) -> List[float]:
        rewards = np.array(rewards).astype(np.float32)
        assert rewards.ndim == 1, "rewards must be a 1D array"
        mean = np.mean(rewards)
        if self.unbiased:
            result = rewards - mean
        else:
            result = (rewards - mean) / (np.std(rewards) + self.eps)

        return result.tolist()

    def ready(self) -> bool:
        return self.reward_fn is not None


_register_rule_based_algo(Algo.GRPO, GRPO)
