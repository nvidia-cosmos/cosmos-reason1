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

from cosmos_reason1.utils.distributed import prevent_vllm_from_setting_nccl_env

prevent_vllm_from_setting_nccl_env()
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.parallelism import ParallelDims
from cosmos_reason1.policy.config import Config as RolloutConfig
from cosmos_reason1.utils.distributed import (
    init_distributed,
    destroy_distributed,
    get_controller_metadata,
)

import sys
from cosmos_reason1.rollout.vllm_rollout.vllm_rollout_worker import vLLMRolloutWorker


def run_rollout():
    ctrl_ip, ctrl_port, metadata = get_controller_metadata()

    if metadata["config"] is None:
        raise RuntimeError(
            f"[Rollout] Please first go to http://{ctrl_ip}:{ctrl_port} to configure training parameters."
        )

    cosmos_rollout_config = RolloutConfig.from_dict(
        metadata["config"]
    )  # just use config as key temporarily

    task_type = cosmos_rollout_config.train.train_policy.type
    if task_type not in ["grpo"]:
        logger.info(
            "[Rollout] Task in controller is not type of Reinforcement Learning. Aborted."
        )
        sys.exit(0)

    logger.info(
        f"[Rollout] Loaded rollout configuration: {cosmos_rollout_config.rollout.key_values()}"
    )

    parallel_dims = ParallelDims.from_config(
        parallesim_config=cosmos_rollout_config.rollout.parallelism
    )

    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    try:
        rollout_worker = vLLMRolloutWorker(cosmos_rollout_config, parallel_dims)
        rollout_worker.work()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        destroy_distributed()
        logger.info("[Rollout] Destroy context of torch dist.")


if __name__ == "__main__":
    run_rollout()
