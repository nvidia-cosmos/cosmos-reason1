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

from enum import IntEnum
from pathlib import Path
import os

CACHE_DIR = Path(
    os.environ.get("COSMOS_CACHE_DIR", Path.home() / ".cache" / "cosmos_reason1")
)

COSMOS_TCP_STORE_TIMEOUT = 10000
COSMOS_ROLLOUT_TRAJECTORY_SIZE = 30

# Heartbeat used to make sure the main thread is alive.
# Mostly, Heartbeat report is non-blocking in a separate thread,
# so we can use a shorter timeout threshold.
COSMOS_HEARTBEAT_TIMEOUT = 200  # 200 seconds
COSMOS_HEARTBEAT_SEND_INTERVAL = 60  # 60 seconds

COSMOS_ROLLOUT_SCAN_INTERVAL = 10  # 10 seconds
COSMOS_ROLLOUT_STEP_INTERVAL = 100  # 100 steps
COSMOS_NCCL_ERROR_CLEAN_REPLICA_DELAY = 10  # 10 seconds


class CosmosHttpRetryConfig:
    max_retries: int = 60
    retries_per_delay: int = 5
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0


COSMOS_HTTP_RETRY_CONFIG = CosmosHttpRetryConfig()
COSMOS_HTTP_LONG_WAIT_MAX_RETRY = 100


class Algo:
    GRPO = "grpo"
    PPO = "ppo"


class RewardFn:
    DIRECT_MATH = "direct_math"
    BOXED_MATH = "boxed_math"
    SINGLE_CHOICE = "single_choice"
    GSM8K = "gsm8k"
    FORMAT = "format"
    OVERLONG = "overlong"

    @classmethod
    def from_string(cls, value: str):
        mapping = {
            "direct_math": cls.DIRECT_MATH,
            "boxed_math": cls.BOXED_MATH,
            "single_choice": cls.SINGLE_CHOICE,
            "gsm8k": cls.GSM8K,
            "format": cls.FORMAT,
            "overlong": cls.OVERLONG,
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
    CMD_READING_TIMEOUT_MS = 10 * 1000  # 10 seconds
    CMD_FETCH_SIZE = 5
    STREAM_MAXLEN = 10000  # Keep latest n message entries
    ROLLOUT_READING_TIMEOUT_MS = 10 * 1000  # 10 seconds
    ROLLOUT_FETCH_SIZE = 8
