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
import torch

from typing import override

import cosmos_reason1._cpp as cosmos_c
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.wandb_logger import init_wandb
from cosmos_reason1.utils.parallelism import ParallelDims
from cosmos_reason1.utils.distributed import (
    init_distributed,
    destroy_distributed,
    get_controller_metadata,
    gradient_reduce_across_dp_replicas_,
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


class mock_GRPOTrainer(GRPOTrainer):
    """
    To compat with vLLMRolloutWorker, we just mock the necessary method.

    1. we try to dump the model's grad before optimize.step()
    """

    dumped_grad_filename_pattern = "grad-\d+-\d+-\d+.pt"

    @override
    def execute_all_reduce(self, command=None):
        """
        reduce necessary grad and dump to file for later check
        """
        for model_part in self.model_parts:
            # Do allreduce of gradient in all policy replicas.
            gradient_reduce_across_dp_replicas_(
                [p for p in model_part.parameters()], self.inter_policy_nccl
            )

            # avoid dp_replicate not scale up
            if self.train_step < 5:
                continue

            # TEST, here we only check embed_token's grad
            for name, obj in model_part.named_parameters():
                if "embed" in name:
                    assert isinstance(
                        obj, torch.distributed.tensor.DTensor
                    ), "Object is not a DTensor"
                    # dump the gradient to disk
                    full_grad = obj.grad.full_tensor().cpu()
                    if full_grad.sum() == 0:
                        logger.info(f"[Policy] Gradient of {name} is zero. Skip dump")
                        continue
                    if self.global_rank != 0:
                        continue
                    inter_nccl_cnt = (
                        1
                        if self.inter_policy_nccl == -1
                        else cosmos_c.get_nccl_comm_count(self.inter_policy_nccl)
                    )
                    torch.save(
                        {name: full_grad},
                        os.path.join(
                            os.path.dirname(self.config.train.output_dir),
                            f"grad_{name}-{self.parallel_dims.dp_shard}-{inter_nccl_cnt}-{os.getpid()}-{self.train_step}.pt",
                        ),
                    )
                    exit(0)

        self.train_stream.wait_stream(self.sync_weight_stream)
        logger.debug(
            f"[Policy] Optimization step {self.optimize_step + 1} at train step {self.train_step + 1} finished."
        )
        # we try freeze the model prarameters, so don't call optimizer.step()
        self.optimize_step += 1
        self.optimizers.zero_grad()
        return True


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
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    if cosmos_config.logging.enable_logging:
        init_wandb(cosmos_config, parallel_dims)

    policy_type = cosmos_config.train.train_policy.type

    try:
        if policy_type == "grpo":
            logger.info("Starting GRPO training...")
            trainer = mock_GRPOTrainer(
                config=cosmos_config, parallel_dims=parallel_dims
            )
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
