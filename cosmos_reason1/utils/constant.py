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

import os
from enum import IntEnum

COSMOS_TCP_STORE_TIMEOUT = 10000
COSMOS_ROLLOUT_TRAJECTORY_SIZE = 30

COSMOS_HEARTBEAT_TIMEOUT = 5 * 60  # 5 minutes
COSMOS_ROLLOUT_SCAN_INTERVAL = 10  # 10 seconds
COSMOS_ROLLOUT_STEP_INTERVAL = 100  # 100 steps
COSMOS_ROLLOUT_PROMPT_QUEUE_MAX_SIZE = 50  # 50 prompts
COSMOS_HEARTBEAT_SEND_INTERVAL = 60  # 60 seconds

# We use this env to indicate rollout worker will
# only do the correctness check after first P2R weight sync
# done and then it will stop.
COSMOS_WEIGHT_SYNC_CHECK = (
    os.getenv("COSMOS_WEIGHT_SYNC_CHECK", "false").lower() == "true"
)


class Algo:
    GRPO = "grpo"
    PPO = "ppo"


class RewardFn:
    BOXED_MATH = "boxed_math"
    SINGLE_CHOICE = "single_choice"
    FORMAT = "format"

    @classmethod
    def from_string(cls, value: str):
        mapping = {
            "boxed_math": cls.BOXED_MATH,
            "single_choice": cls.SINGLE_CHOICE,
            "format": cls.FORMAT,
        }
        if value not in mapping:
            raise ValueError(f"Invalid value: {value}")
        return mapping[value]


class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001
    # Added for Vision API
    INVALID_IMAGE = 40002
    ALREADY_EXISTS = 40003

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303
    INVALID_REQUEST = 400304

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    REQUEST_CANCELLED = 49901

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004

    SERVICE_UNAVAILABLE = 50301


class RedisStreamConstant:
    CMD_READING_TIMEOUT_MS = 0  # `0` means never timeout to prevent frequent polling
    CMD_FETCH_SIZE = 5
    STREAM_MAXLEN = 10000  # Keep latest n message entries
    ROLLOUT_READING_TIMEOUT_MS = (
        0  # `0` means never timeout to prevent frequent polling
    )
    ROLLOUT_FETCH_SIZE = 8
