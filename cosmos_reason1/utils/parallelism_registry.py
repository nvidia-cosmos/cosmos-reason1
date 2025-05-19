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

from typing import Dict, Callable


_MODEL_WEIGHT_PARALLELISM_REGISTRY_POLICY: Dict[str, Callable] = {}
_MODEL_WEIGHT_PARALLELISM_REGISTRY_ROLLOUT: Dict[str, Callable] = {}


def register_policy_parallelism_strategy(reg_key: str, *, allow_override: bool = False):
    def decorator(func: Callable) -> Callable:
        if not allow_override and reg_key in _MODEL_WEIGHT_PARALLELISM_REGISTRY_POLICY:
            raise ValueError(f"Function '{reg_key}' is already registered.")
        _MODEL_WEIGHT_PARALLELISM_REGISTRY_POLICY[reg_key] = func
        return func

    return decorator


def register_rollout_parallelism_strategy(
    reg_key: str, *, allow_override: bool = False
):
    def decorator(func: Callable) -> Callable:
        if not allow_override and reg_key in _MODEL_WEIGHT_PARALLELISM_REGISTRY_ROLLOUT:
            raise ValueError(f"Function '{reg_key}' is already registered.")
        _MODEL_WEIGHT_PARALLELISM_REGISTRY_ROLLOUT[reg_key] = func
        return func

    return decorator


def register_parallelism_strategy(reg_key: str, *, allow_override: bool = False):
    def decorator(func: Callable) -> Callable:
        if not allow_override and reg_key in _MODEL_WEIGHT_PARALLELISM_REGISTRY_POLICY:
            raise ValueError(f"Function '{reg_key}' is already registered.")
        _MODEL_WEIGHT_PARALLELISM_REGISTRY_POLICY[reg_key] = func
        if not allow_override and reg_key in _MODEL_WEIGHT_PARALLELISM_REGISTRY_ROLLOUT:
            raise ValueError(f"Function '{reg_key}' is already registered.")
        _MODEL_WEIGHT_PARALLELISM_REGISTRY_ROLLOUT[reg_key] = func
        return func

    return decorator


def get_policy_parallelism_strategy(model_name: str) -> Callable:
    return _MODEL_WEIGHT_PARALLELISM_REGISTRY_POLICY[model_name]


def get_rollout_parallelism_strategy(model_name: str) -> Callable:
    return _MODEL_WEIGHT_PARALLELISM_REGISTRY_ROLLOUT[model_name]
