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

import time
import math
from queue import Queue
from strenum import StrEnum
from typing import Dict, List, Iterator, Any, Optional
from torch.utils.data import DataLoader
from cosmos_reason1.utils.constant import COSMOS_HEARTBEAT_TIMEOUT
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.policy.config import Config
from cosmos_reason1.dispatcher.replica import Replica, Atom, Rollout
from cosmos_reason1.dispatcher.protocol import Role
import cosmos_reason1.dispatcher.command as command
from cosmos_reason1.utils.redis_stream import RedisStreamHandler
from cosmos_reason1.utils.wandb_logger import (
    is_wandb_available,
    log_wandb,
)
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm


class PolicyStatus(StrEnum):
    """
    Enum for policy status.
    There are 7 statuses:
    UNINITIALIZED: The policy is uninitialized.
    READY: The policy is ready to run.
    RUNNING: The policy is running.
    REDUCED: The policy has finished reduce.
    END: The policy has finished.
    """

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    RUNNING = "running"
    REDUCED = "reduced"
    END = "end"


class PolicyStatusManager:
    """
    A class to manage the status of a policy.
    """

    policy_replicas: Dict[str, Replica]
    policy_init_done: bool = False

    # Global status
    remain_samples_num: int
    current_step: int
    total_steps: int

    # Instance status
    status: Dict[str, PolicyStatus]

    def __init__(self):
        self.policy_replicas = {}
        self.total_steps = 0
        self.current_step = 0
        self.rollout_buffer = Queue()
        self.remain_samples_num = 0
        self.status = {}

        # Validation
        self.val_iters: Dict[int, Iterator] = {}
        self.activated_val_iter: Optional[Iterator] = None
        self.val_report_data: Dict[int, List[Any]] = {}

    def setup(
        self,
        config: Config,
        redis_handler: RedisStreamHandler,
        remain_samples_num: int,
        tokenizer: AutoTokenizer,
        val_dataloader: Optional[DataLoader] = None,
    ):
        self.redis_handler = redis_handler
        self.config = config
        self.remain_samples_num = remain_samples_num
        self.tokenizer = tokenizer
        self.val_dataloader = val_dataloader

    def __len__(self) -> int:
        """
        Get the number of policies.
        """
        return len(self.policy_replicas)

    def __iter__(self) -> Iterator[Replica]:
        """
        Iterate over the policy replicas.
        """
        for replica in sorted(self.policy_replicas.values(), key=lambda x: x.name):
            yield replica

    def __contains__(self, replica_name: str) -> bool:
        """
        Check if the replica is in the status manager.
        """
        return replica_name in self.policy_replicas

    def __getitem__(self, replica_name: str) -> Replica:
        """
        Get the replica from the status manager.
        """
        return self.policy_replicas.get(replica_name)

    def training_finished(self) -> bool:
        """
        Check if the training is finished.
        """
        return self.current_step >= self.total_steps and self.total_steps > 0

    def maintain_life_status(self):
        """
        Maintain the life status of the rollout.
        """
        dead_replicas = set()
        now = time.time()
        for replica in self:
            if now - replica.status.heartbeat_timestamp > COSMOS_HEARTBEAT_TIMEOUT:
                logger.warning(f"[Controller] Policy {replica.name} is dead")
                dead_replicas.add(replica.name)
        for replica_name in dead_replicas:
            self.unregister(replica_name)

    def set_status(self, name: str, status: PolicyStatus):
        """
        Set the status of the policy.
        """
        if name not in self.status:
            assert (
                status == PolicyStatus.UNINITIALIZED
            ), "Policy status should be UNINITIALIZED when first created"
            self.status[name] = status
            return
        assert (
            status != PolicyStatus.UNINITIALIZED
        ), "Policy status should not be UNINITIALIZED when already created"
        self.status[name] = status

    def recompute_total_steps(
        self, explicit_num_remaining_samples: Optional[int] = None
    ):
        """
        Set the ranks of the policies.
        """
        # Update total step when policy replicas are set
        num_policy_replicas = len(self.get_all_atoms_arrived_replicas())
        if num_policy_replicas > 0:
            num_remaining_samples = (
                explicit_num_remaining_samples or self.remain_samples_num
            )
            self.total_steps = self.current_step + num_remaining_samples // (
                self.config.train.train_batch_per_replica * num_policy_replicas
            )

    def get_status(self, name: str) -> PolicyStatus:
        """
        Get the status of the policy.
        """
        if name not in self.status:
            raise KeyError(f"Policy {name} not found")
        return self.status[name]

    def all_with_status(self, status: List[PolicyStatus]) -> bool:
        """
        Check if all policies have the given status.
        """
        return all([x in status for x in self.status.values()])

    def all_reduced(self) -> bool:
        """
        Check if all policies are reduced.
        """
        return self.all_with_status([PolicyStatus.REDUCED])

    def all_ready(self) -> bool:
        """
        Check if all policies are ready.
        """
        return self.all_with_status([PolicyStatus.READY])

    def all_ready_or_reduced(self) -> bool:
        """
        Check if all policies are ready or reduced.
        """
        return self.all_with_status([PolicyStatus.READY, PolicyStatus.REDUCED])

    def set_ncclerror(self, replica_name: str, timestamp: int):
        """
        Set the timeout ack of the policy.
        """
        self[replica_name].status.nccl_error_timestamp = timestamp

    def clear_ncclerror(self):
        """
        Clear the timeout ack of the policy.
        """
        for replica in self:
            replica.status.nccl_error_timestamp = None

    def get_all_policy_report_ncclerror(self) -> Dict[str, int]:
        """
        Get all the timeout ack of the policies.
        """
        return {
            replica.name: replica.status.nccl_error_timestamp
            for replica in self
            if replica.status.nccl_error_timestamp is not None
        }

    def heartbeat(self, replica_name: str):
        timestamp: int = int(time.time())
        if replica_name not in self:
            logger.warning(
                f"[Controller] Replica {replica_name} not found in policy status manager."
            )
            return
        self[replica_name].status.heartbeat_timestamp = timestamp

    def shutdown(self):
        """
        Shutdown the status manager.
        """
        self.policy_init_done = False

    def unregister(self, replica_name: str):
        """
        Unregister the replica from the status manager.
        """
        assert (
            replica_name in self
        ), f"Replica {replica_name} not found in policy status manager"

        replica = self.policy_replicas.pop(replica_name)

        if self.training_finished():
            # This policy replica is normally finished
            # Do not trigger rebuild mesh since everything is gonna be finished shortly
            logger.info(f"[Controller] Replica {replica_name} is stopping.")
            return

        valid_replicas = self.get_all_atoms_arrived_replicas()
        if replica.in_mesh and len(valid_replicas) > 0:
            self.trigger_rebuild_mesh(valid_replicas)

    def register(
        self,
        atom: Atom,
        config: Config,
        rollout_status_manager: "RolloutStatusManager",
        **kwargs,
    ):
        """
        Register the atom to the status manager.
        """
        replica = self[atom.replica_name]
        if replica is None:
            replica = Replica(atom.replica_name, Role.POLICY, [atom])
            self.policy_replicas[atom.replica_name] = replica
        else:
            replica.arrive(atom)
        atom.bind_replica(replica)
        current_policy_replica = replica

        # post register hook
        if not self.policy_init_done:
            if len(self.policy_replicas) > config.policy.parallelism.n_init_replicas:
                config.policy.parallelism.n_init_replicas = len(self.policy_replicas)
                logger.info(
                    f"[Controller] Update policy n_init_replicas to {config.policy.parallelism.n_init_replicas} replicas"
                )

        # Check if all atoms of the replica have arrived
        if replica.all_atoms_arrived:
            if replica.start_time == -1:
                replica.start_time = int(time.time())
            logger.info(
                f"[Controller] All atoms of {Role.POLICY} Replica {replica.name} has been set."
            )
            self.set_status(replica.name, PolicyStatus.UNINITIALIZED)
            # Check total valid policy replicas
            valid_replicas = []
            if not hasattr(self, "policy_atoms_in_replica"):
                self.policy_atoms_in_replica = int(math.prod(atom.group_size))

            for r in self.policy_replicas.values():
                if r.all_atoms_arrived:
                    valid_replicas.append(r)

            # Load weight for the first loaded replica policy
            if len(valid_replicas) == 1:
                assert not hasattr(
                    self, "_first_policy_replica_arrived"
                ), "Expect only one policy replica to load weight during training process"
                self._first_policy_replica_arrived = True
                # This is the first policy replica to arrive, it is responsible for weight initialization
                command.WeightResumeCommand.trigger(
                    current_policy_replica, redis_handler=self.redis_handler
                )

                # Check whether there is any valid rollout replicas
                any_valid_rollout_replica = None
                sorted_rollout_replicas = sorted(
                    rollout_status_manager.rollout_replicas.values(),
                    key=lambda x: x.start_time,
                )
                for r in sorted_rollout_replicas:
                    if r.all_atoms_arrived:
                        any_valid_rollout_replica = r
                        break
                if any_valid_rollout_replica:
                    command.PolicyToRolloutUnicastCommand.trigger(
                        src_replica=current_policy_replica,
                        dst_replica=any_valid_rollout_replica,
                        src_replica_size=self.policy_atoms_in_replica,
                        dst_replica_size=rollout_status_manager.rollout_atoms_in_replica,
                        weight_step=None,
                        total_steps=None,
                        redis_handler=self.redis_handler,
                    )
                    logger.info(
                        f"[Controller] Trigger PolicyToRolloutUnicastCommand to {any_valid_rollout_replica.name} via Policy registration"
                    )
                else:
                    logger.info(
                        "[Controller] No valid rollout replicas found, skip PolicyToRolloutUnicastCommand"
                    )
            self.post_register_hook(
                valid_replicas,
                atom.replica,
                config,
                rollout_status_manager,
            )
        return replica

    def trigger_rebuild_mesh(self, valid_replicas: List[Replica]):
        # Always tell the policy to rebuild mesh even there is only one policy replica
        command.BuildMeshCommand.trigger(
            valid_replicas, redis_handler=self.redis_handler
        )
        self.recompute_total_steps()

    def post_register_hook(
        self,
        valid_replicas: List[Replica],
        target_replica: Replica,
        config: Config,
        rollout_status_manager: "RolloutStatusManager",
    ):
        sorted_valid_replicas = sorted(valid_replicas, key=lambda x: x.start_time)

        if (
            not self.policy_init_done
            and len(valid_replicas) >= config.policy.parallelism.n_init_replicas
        ):
            # This is the case when all required replicas have arrived

            self.policy_init_done = True
            # Trigger mesh building (Typically only occurs during initialization)

            # we need buildmesh, event there is only one replica. (trigger HANccl buildmesh)
            # 1. Trigger mesh building
            self.trigger_rebuild_mesh(valid_replicas)

            # 2. Trigger weight/optimizer state synchronization
            if len(valid_replicas) > 1:
                # Only broadcast when there are multiple policy replicas
                initialized_replica = None
                for replica in sorted_valid_replicas:
                    # We will select the first replica that has weights loaded in view of command
                    if (
                        replica.weights_loaded_in_view_of_command
                        and replica in valid_replicas
                    ):
                        initialized_replica = replica
                        break
                assert (
                    initialized_replica is not None
                ), "No replica was selected to load weights"
                command.PolicyToPolicyBroadcastCommand.trigger(
                    src_replica=initialized_replica,
                    dst_replicas=valid_replicas,
                    redis_handler=self.redis_handler,
                )
            # Set all policy replicas to `ready`
            for replica in valid_replicas:
                self.set_status(replica.name, PolicyStatus.READY)
        elif (
            not self.policy_init_done
            and len(valid_replicas) < config.policy.parallelism.n_init_replicas
        ):
            # This is the case when replicas are in the initialization stage
            logger.info(
                f"Waiting for {config.policy.parallelism.n_init_replicas - len(valid_replicas)} more replicas to arrive"
            )
        else:
            # This is the case when the dynamic scaling is triggered
            assert (
                self.policy_init_done
            ), "Policy initialization must be done before building another mesh"

            assert (
                target_replica.status.mesh_rank == -1
            ), "Target replica should not be in the mesh"

            # This occurs when new dynamic scaling is triggered
            initialized_replica = None
            for replica in sorted_valid_replicas:
                if (
                    replica.weights_loaded_in_view_of_command
                    and replica in valid_replicas
                ):
                    # We will select the first replica that has weights loaded in view of command
                    # to broadcast weights
                    initialized_replica = replica
                    break
            assert (
                initialized_replica is not None
            ), "No replica was selected to load weights"
            self.trigger_rebuild_mesh(valid_replicas)

            command.PolicyToPolicyUnicastCommand.trigger(
                src_replica=initialized_replica,
                dst_replica=target_replica,
                redis_handler=self.redis_handler,
            )
            self.set_status(target_replica.name, PolicyStatus.READY)

    ############################################################
    # utility functions
    ############################################################
    def validation_activate_dataloader(self, validation_step: int):
        if validation_step not in self.val_iters:
            logger.info(
                f"[Controller] Activating validation dataloader for step {validation_step}, with length {len(self.val_dataloader)}"
            )
            self.val_iters[validation_step] = iter(self.val_dataloader)
            self.activated_val_iter = self.val_iters[validation_step]
            self.activated_val_tqdm = tqdm(
                desc="validation",
                total=len(self.val_dataloader),
            )

    def validation_get_dataloader(
        self, validation_step: Optional[int] = None
    ) -> Iterator:
        if validation_step is None:
            return self.activated_val_iter
        else:
            return self.val_iters[validation_step]

    def validation_report_validation_results(
        self,
        validation_step: int,
        validation_results: List[List[Rollout]],
        rollout_status_manager: "RolloutStatusManager",
    ):
        if validation_step not in self.val_report_data:
            self.val_report_data[validation_step] = []

        self.val_report_data[validation_step].extend(validation_results)
        num_rollout_replicas = len(
            rollout_status_manager.get_all_atoms_arrived_replicas()
        )
        n_items_of_this_step = sum(
            len(x) for x in self.val_report_data[validation_step]
        )
        validation_finished = (
            len(self.val_report_data[validation_step]) == num_rollout_replicas
        )
        validation_finished = validation_finished or n_items_of_this_step == len(
            self.val_dataloader
        )

        if self.activated_val_tqdm:
            self.activated_val_tqdm.update(n_items_of_this_step)
        else:
            logger.error("[Controller] Validation tqdm is not activated")

        # Check if all rollout replicas have reported validation results
        if validation_finished:
            # Validation is finished, trigger next step training
            self.activated_val_iter = None
            self.activated_val_tqdm.clear()
            self.activated_val_tqdm = None

            try:
                all_rollouts_lists: List[List[Rollout]] = self.val_report_data[
                    validation_step
                ]
                if all_rollouts_lists:
                    rewards = []
                    for rollouts in all_rollouts_lists:
                        rewards.extend([r.reward for r in rollouts])
                    avg_reward = np.mean(rewards)
                    std_reward = np.std(rewards)
                    max_reward = np.max(rewards)
                    min_reward = np.min(rewards)

                    report_data = {
                        "val/reward_avg": avg_reward,
                        "val/reward_std": std_reward,
                        "val/reward_max": max_reward,
                        "val/reward_min": min_reward,
                        "val/rollout_count": len(rewards),
                        "val/step": validation_step,
                    }
                    logger.info(
                        f"[Controller] Validation finished, average reward: {avg_reward}, total rollouts: {len(rewards)}, max reward: {max_reward}, min reward: {min_reward}, std reward: {std_reward} at step {validation_step}"
                    )
                    if "wandb" in self.config.logging.logger and is_wandb_available():
                        log_wandb(
                            data=report_data,
                            step=validation_step,
                        )
            except Exception as e:
                logger.error(f"[Controller] Error reporting validation results: {e}")

            # The order is important, because the previous code block logs the previous step's validation results
            # while `try_trigger_data_fetch_and_training` will immediately report the next step's results
            self.try_trigger_data_fetch_and_training()

    def total_pending_rollouts(self) -> int:
        """
        Get the total pending rollouts.
        """
        return self.rollout_buffer.qsize()

    def get_all_atoms_arrived_replicas(self) -> List[Replica]:
        """
        Get all the replicas that have all atoms arrived.
        """
        return [
            replica
            for replica in self.policy_replicas.values()
            if replica.all_atoms_arrived
        ]

    def put_rollout(self, rollout: Rollout):
        """
        Dispatch the rollout to the policy replicas in a round-robin manner.
        It is that replica's responsibility to dispatch the rollout to further (DP_SHARD) atoms.
        """
        if self.config.rollout.include_stop_str_in_output:
            if self.tokenizer.eos_token is not None and rollout.completion is not None:
                if not rollout.completion.endswith(self.tokenizer.eos_token):
                    rollout.completion = rollout.completion + self.tokenizer.eos_token
        self.rollout_buffer.put(rollout)
        self.try_trigger_data_fetch_and_training()

    def train_ack(
        self,
        replica_name: str,
        step: int,
        profile_finished: bool,
        report_data: Dict[str, Any],
        rollout_status_manager: "RolloutStatusManager",
    ):
        if replica_name not in self:
            raise Exception(f"Replica {replica_name} not found")

        self.set_status(replica_name, PolicyStatus.REDUCED)

        if not hasattr(self, "report_data_list"):
            self.report_data_list = []
        self.report_data_list.append(report_data)
        if self.all_reduced():
            # All replicas have been reduced, trigger allreduce
            need_sync_weight = step % self.config.train.sync_weight_interval == 0
            # If the current step is the last step, we need to sync weight always to act as ending signal
            need_sync_weight = need_sync_weight or step == self.total_steps
            # If validation is enabled, we need to sync weight every validation step
            if self.config.train.enable_validation:
                need_sync_weight = need_sync_weight or (
                    step % self.config.train.validation_step == 0
                )

            if profile_finished:
                # Only reset the do_profile flag if the profile is finished
                logger.debug(f"[Controller] Unset the profile mode of {replica_name}")
                self[replica_name].sub_profiler_config.do_profile = False

            # Sum and report data
            if self.config.logging.logger:
                total_loss_avg = np.mean(
                    [data["train/loss_avg"] for data in self.report_data_list]
                )
                total_loss_max = np.max(
                    [data["train/loss_max"] for data in self.report_data_list]
                )
                total_learning_rate = self.report_data_list[0]["train/learning_rate"]
                train_step = self.report_data_list[0]["train_step"]
                total_steps = self.total_steps
                if train_step > 1:
                    total_iter_time_avg = np.mean(
                        [data["train/iteration_time"] for data in self.report_data_list]
                    )
                self.report_data_list = []

                if "wandb" in self.config.logging.logger and is_wandb_available():
                    if train_step > 1:
                        log_wandb(
                            data={"train/pre_iteration_time": total_iter_time_avg},
                            step=train_step,
                        )

                    log_wandb(
                        data={
                            "train/loss_avg": total_loss_avg,
                            "train/loss_max": total_loss_max,
                            "train/learning_rate": total_learning_rate,
                        },
                        step=train_step,
                    )
                if "console" in self.config.logging.logger:
                    if train_step > 1:
                        logger.debug(
                            f"Step: {train_step-1}/{total_steps}, Iteration time: {total_iter_time_avg:2f} s"
                        )
                    logger.info(
                        f"Step: {train_step}/{total_steps}, Average loss: {total_loss_avg:.5f}, Max loss: {total_loss_max:.5f}, Learning rate: {total_learning_rate:.5e}."
                    )

            # All replicas have been reduced, trigger weight sync
            any_loaded_replica = None
            sorted_replicas = sorted(
                self.get_all_atoms_arrived_replicas(), key=lambda x: x.start_time
            )
            for replica in sorted_replicas:
                if any_loaded_replica is None:
                    any_loaded_replica = replica
                self.set_status(replica.name, PolicyStatus.READY)

            # P->R & R->R
            if need_sync_weight:
                self.trigger_weight_sync(any_loaded_replica, rollout_status_manager)
            # Trigger next step training if data is available
            self.try_trigger_data_fetch_and_training()

    def trigger_weight_sync(
        self, policy_replica: Replica, rollout_status_manager: "RolloutStatusManager"
    ):
        any_loaded_rollout_replica = None
        valid_rollout_replicas = []
        sorted_replicas = sorted(
            rollout_status_manager.get_all_atoms_arrived_replicas(),
            key=lambda x: x.start_time,
        )
        for rollout_replica in sorted_replicas:
            if any_loaded_rollout_replica is None:
                any_loaded_rollout_replica = rollout_replica
            valid_rollout_replicas.append(rollout_replica)
        if any_loaded_rollout_replica is None:
            return
        command.PolicyToRolloutUnicastCommand.trigger(
            src_replica=policy_replica,
            dst_replica=any_loaded_rollout_replica,
            src_replica_size=self.policy_atoms_in_replica,
            dst_replica_size=rollout_status_manager.rollout_atoms_in_replica,
            weight_step=self.current_step,
            total_steps=self.total_steps,
            redis_handler=self.redis_handler,
        )

        command.RolloutToRolloutBroadcastCommand.trigger(
            src_replica=any_loaded_rollout_replica,
            dst_replicas=valid_rollout_replicas,
            weight_step=self.current_step,
            total_steps=self.total_steps,
            redis_handler=self.redis_handler,
        )

    def try_trigger_data_fetch_and_training(self, is_fake_last_cmd=False):
        # If the validation dataloader is activated, do not trigger data fetch and training
        if self.activated_val_iter is not None:
            return

        arrived_replicas = self.get_all_atoms_arrived_replicas()
        # no replicas arrived, do nothing
        if len(arrived_replicas) == 0:
            return

        if is_fake_last_cmd:
            required_rollouts = 0
            all_ready_or_reduced = True
            items_count = 0
            assert (
                self.current_step + 1 == self.total_steps
            ), "The last command should be fake and next step should be the last step"
        else:
            items_count = self.config.train.train_batch_per_replica
            required_rollouts = items_count * len(arrived_replicas)
            all_ready_or_reduced = self.all_ready_or_reduced() and (
                self.rollout_buffer.qsize() >= required_rollouts
            )

        # If the last command is fake, we need to trigger data fetch and training no matter
        # whether there are enough rollouts or whether replicas are `ready` or `reduced`.
        if all_ready_or_reduced:
            rollouts_of_this_step: List[Rollout] = []
            self.remain_samples_num -= required_rollouts

            # From controller's perspective, the training step is already increased
            self.current_step += 1

            if self.config.train.enable_validation and (
                self.current_step % self.config.train.validation_step == 0
                or self.current_step == self.total_steps
            ):
                self.validation_activate_dataloader(self.current_step)

            for replica in arrived_replicas:
                # Dispatch rollouts to policy replicas
                for _ in range(items_count):
                    rollout = self.rollout_buffer.get()
                    replica.put_rollout(rollout, self.redis_handler)
                    rollouts_of_this_step.append(rollout)
                command.DataFetchCommand.trigger(
                    replica=replica,
                    items_count=items_count,
                    global_step=self.current_step,
                    total_steps=self.total_steps,
                    # `remain_samples_num` is just for checkpointing the training progress
                    remain_samples_num=self.remain_samples_num,
                    redis_handler=self.redis_handler,
                )
                self.set_status(replica.name, PolicyStatus.RUNNING)

            # Report the reward, length, etc.
            # These properties are already ready to be reported before being trained
            if self.config.logging.logger and rollouts_of_this_step:
                rewards = []
                completion_lengths = []
                for rollout in rollouts_of_this_step:
                    rewards.append(rollout.reward)
                    completion_lengths.append(
                        len(self.tokenizer.encode(rollout.completion))
                    )
                report_data = {
                    "train/reward_mean": np.mean(rewards),
                    "train/reward_std": np.std(rewards),
                    "train/reward_max": np.max(rewards),
                    "train/reward_min": np.min(rewards),
                    "train/completion_length_mean": np.mean(completion_lengths),
                    "train/completion_length_max": np.max(completion_lengths),
                }

                if "wandb" in self.config.logging.logger and is_wandb_available():
                    log_wandb(report_data, step=self.current_step)

                if "console" in self.config.logging.logger:
                    logger.info(
                        f"Step: {self.current_step}/{self.total_steps}, Reward Mean: {report_data['train/reward_mean']:.4f}, Reward Std: {report_data['train/reward_std']:.4f}, Reward Max: {report_data['train/reward_max']:.4f}, Reward Min: {report_data['train/reward_min']:.4f}, Completion Length Mean: {report_data['train/completion_length_mean']:.2f}, Completion Length Max: {report_data['train/completion_length_max']:.2f}"
                    )


class RolloutStatusManager:
    """
    A class to manage the status of rollout replicas.
    """

    rollout_replicas: Dict[str, Replica]
    rollout_init_done: bool

    def __init__(self):
        self.rollout_replicas = {}
        self.rollout_init_done = False

    def setup(
        self,
        config: Config,
        redis_handler: RedisStreamHandler,
        tokenizer: AutoTokenizer,
    ):
        self.redis_handler = redis_handler
        self.config = config
        self.tokenizer = tokenizer
        """
        Maintain the life status of the policy and rollout replicas.
        """
        return len(self.rollout_replicas)

    def __len__(self) -> int:
        """
        Get the number of rollout replicas.
        """
        return len(self.rollout_replicas)

    def __iter__(self) -> Iterator[Replica]:
        """
        Iterate over the policy replicas.
        """
        for replica in sorted(self.rollout_replicas.values(), key=lambda x: x.name):
            yield replica

    def __contains__(self, replica_name: str) -> bool:
        """
        Check if the replica is in the status manager.
        """
        return replica_name in self.rollout_replicas

    def __getitem__(self, replica_name: str) -> Replica:
        """
        Get the replica from the status manager.
        """
        return self.rollout_replicas.get(replica_name)

    def maintain_life_status(self):
        """
        Maintain the life status of the rollout.
        """
        now = time.time()
        dead_replicas = set()
        for replica in self:
            if now - replica.status.heartbeat_timestamp > COSMOS_HEARTBEAT_TIMEOUT:
                logger.warning(f"[Controller] Rollout {replica.name} is dead")
                dead_replicas.add(replica.name)
        for replica_name in dead_replicas:
            self.unregister(replica_name)

    def heartbeat(self, replica_name: str):
        timestamp: int = int(time.time())
        if replica_name not in self:
            logger.warning(
                f"[Controller] Replica {replica_name} not found in both policy and rollout."
            )
            return
        self[replica_name].status.heartbeat_timestamp = timestamp

    ############################################################
    # utility functions
    ############################################################
    def get_all_atoms_arrived_replicas(self) -> List[Replica]:
        """
        Get all the replicas that have all atoms arrived.
        """
        return [
            replica
            for replica in self.rollout_replicas.values()
            if replica.all_atoms_arrived
        ]

    def unregister(self, replica_name: str, policy_status_manager: PolicyStatusManager):
        """
        Unregister the replica from the status manager.
        """
        assert (
            replica_name in self
        ), f"Replica {replica_name} not found in policy status manager"

        replica = self.rollout_replicas.pop(replica_name)

        if policy_status_manager.training_finished():
            # This policy replica is normally finished
            # Do not trigger rebuild mesh since everything is gonna be finished shortly
            logger.info(f"[Controller] Replica {replica_name} is stopping.")
            return

        if replica.in_mesh and len(self.rollout_replicas) > 0:
            self.trigger_rebuild_mesh(self.get_all_atoms_arrived_replicas())

    def register(
        self,
        atom: Atom,
        config: Config,
        policy_status_manager: PolicyStatusManager,
        **kwargs,
    ):
        """
        Register the atom to the status manager.
        """
        replica = self[atom.replica_name]
        if replica is None:
            replica = Replica(atom.replica_name, Role.ROLLOUT, [atom])
            self.rollout_replicas[atom.replica_name] = replica
        else:
            replica.arrive(atom)
        atom.bind_replica(replica)

        # post register hook
        if not self.rollout_init_done:
            if len(self.rollout_replicas) > config.rollout.parallelism.n_init_replicas:
                config.rollout.parallelism.n_init_replicas = len(self.rollout_replicas)
                logger.info(
                    f"[Controller] Update rollout n_init_replicas to {config.rollout.parallelism.n_init_replicas} replicas"
                )

        # Check if all atoms of the replica have arrived
        if replica.all_atoms_arrived:
            if replica.start_time == -1:
                replica.start_time = int(time.time())
            logger.info(
                f"[Controller] All atoms of {Role.ROLLOUT} Replica {replica.name} has been set."
            )
            # Check total valid rollout replicas
            valid_replicas = []
            if not hasattr(self, "rollout_atoms_in_replica"):
                self.rollout_atoms_in_replica = int(math.prod(atom.group_size))
            for replica in self.rollout_replicas.values():
                if replica.all_atoms_arrived:
                    valid_replicas.append(replica)
            self.post_register_hook(
                valid_replicas,
                atom.replica,
                config,
                policy_status_manager,
            )
        return replica

    def rollout_end(self, replica_name: str):
        """
        Rollout end event.
        """
        replica = self[replica_name]
        if replica is None:
            logger.warning(
                f"[Controller] Rollout {replica_name} not found in RolloutStatusManager"
            )
            return
        replica.status.ended = True

    def all_rollouts_ended(self) -> bool:
        """
        Check if all rollouts have ended.
        """
        return all([replica.status.ended for replica in self.rollout_replicas.values()])

    def trigger_rebuild_mesh(
        self,
        valid_replicas: List[Replica],
    ):
        command.BuildMeshCommand.trigger(
            valid_replicas, redis_handler=self.redis_handler
        )

    def post_register_hook(
        self,
        valid_replicas: List[Replica],
        target_replica: Replica,
        config: Config,
        policy_status_manager: PolicyStatusManager,
    ):
        assert target_replica in valid_replicas
        any_loaded_policy_replica = None
        sorted_valid_policy_replicas = sorted(
            [r for r in policy_status_manager], key=lambda x: x.start_time
        )
        for replica in sorted_valid_policy_replicas:
            if replica.weights_loaded_in_view_of_command:
                # We will select the first replica that has weights loaded in view of command
                # to broadcast weights
                any_loaded_policy_replica = replica
                break
        if any_loaded_policy_replica is None:
            logger.info(
                "No weight-loaded policy replica exists, will be rescheduled after first policy replica is loaded"
            )
            return

        # First P->R Unicast if the policy is ready and all rollout replicas are not ready
        if all(
            [
                not replica.weights_loaded_in_view_of_command
                for replica in valid_replicas
            ]
        ):
            command.PolicyToRolloutUnicastCommand.trigger(
                src_replica=any_loaded_policy_replica,
                dst_replica=target_replica,
                src_replica_size=policy_status_manager.policy_atoms_in_replica,
                dst_replica_size=self.rollout_atoms_in_replica,
                weight_step=None,
                total_steps=None,
                redis_handler=self.redis_handler,
            )
            logger.info(
                f"[Controller] Trigger PolicyToRolloutUnicastCommand to {target_replica.name} via Rollout registration"
            )
        else:
            logger.info(
                "[Controller] No valid policy replicas found in Rollout registration, skip PolicyToRolloutUnicastCommand"
            )

        was_already_initialized = self.rollout_init_done

        if (
            not was_already_initialized
            and len(valid_replicas) == config.rollout.parallelism.n_init_replicas
        ):
            self.rollout_init_done = True
            self.trigger_rebuild_mesh(valid_replicas)

            # ONLY ONCE PER LIFE CYCLE
            # Trigger RolloutToRolloutBroadcastCommand only once after all initial rollout replicas are loaded
            any_loaded_rollout_replica = None
            sorted_valid_replicas = sorted(valid_replicas, key=lambda x: x.start_time)
            for replica in sorted_valid_replicas:
                if (
                    replica.weights_loaded_in_view_of_command
                    and replica in valid_replicas
                ):
                    # We will select the first replica that has weights loaded in view of command
                    # to broadcast weights
                    any_loaded_rollout_replica = replica
                    break
            assert any_loaded_rollout_replica is not None
            command.RolloutToRolloutBroadcastCommand.trigger(
                src_replica=any_loaded_rollout_replica,
                dst_replicas=valid_replicas,
                weight_step=None,
                total_steps=None,
                redis_handler=self.redis_handler,
            )

        elif not self.rollout_init_done:
            assert len(valid_replicas) < config.rollout.parallelism.n_init_replicas
            logger.info(
                f"Waiting for {config.rollout.parallelism.n_init_replicas - len(valid_replicas)} more replicas to arrive"
            )
        else:
            # Dynamic mesh building, no matter what the length of valid_replicas is,
            # we will always trigger mesh building if there are more than one rollout replicas
            self.trigger_rebuild_mesh(valid_replicas)
