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

from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.wandb_logger import is_wandb_available, init_wandb
from cosmos_reason1.utils.parallelism import ParallelDims
from cosmos_reason1.utils.distributed import (
    init_distributed,
    destroy_distributed,
    get_controller_metadata,
)
from cosmos_reason1.policy.trainer.sft_trainer import SFTTrainer
from cosmos_reason1.policy.trainer.grpo_trainer import GRPOTrainer
from cosmos_reason1.policy.config import Config as PolicyConfig

try:
    # for policy and rollout nccl env consistency
    import vllm  # noqa: F401
except ImportError:
    logger.warning("vllm is not installed, skipping nccl env consistency check")
    pass


def run_train():
    ctrl_ip, ctrl_port, metadata = get_controller_metadata()

    if metadata["config"] is None:
        raise RuntimeError(
            f"[Policy] Please first go to http://{ctrl_ip}:{ctrl_port} to configure training parameters."
        )

    cosmos_config = PolicyConfig.from_dict(metadata["config"])
    logger.info(f"[Policy] Loaded configuration: {cosmos_config.key_values()}")

    parallel_dims = ParallelDims.from_config(
        parallesim_config=cosmos_config.policy.parallelism
    )
    init_distributed(cpu_enabled=cosmos_config.train.fsdp_offload)
    parallel_dims.build_mesh(device_type="cuda")

    if cosmos_config.logging.enable_logging:
        if is_wandb_available():
            init_wandb(cosmos_config, parallel_dims)
        else:
            logger.warning(
                "Wandb is not available. Please install it to use wandb logging features."
            )

    policy_type = cosmos_config.train.train_policy.type

    try:
        if policy_type == "grpo":
            logger.info("Starting GRPO training...")
            trainer = GRPOTrainer(config=cosmos_config, parallel_dims=parallel_dims)
            trainer.main_loop()
        elif policy_type == "sft":
            logger.info("Starting SFT training...")
            trainer = SFTTrainer(config=cosmos_config, parallel_dims=parallel_dims)
            trainer.train()
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e
    finally:
        destroy_distributed()
        logger.info("Process group destroyed.")


if __name__ == "__main__":
    run_train()
