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

from cosmos_reason1.dispatcher.replica import Atom, Replica, Rollout
from cosmos_reason1.dispatcher.protocol import Role, MESH_NAMES
from typing import List, Dict, Tuple, Any
import copy
import math
import torch
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.wandb_logger import is_wandb_available, init_wandb
import cosmos_reason1.utils.util as util
import cosmos_reason1.utils.constant as constant
from queue import Queue
from cosmos_reason1.dispatcher.algo.base import REGISTERED_ALGOs
from cosmos_reason1.dispatcher.algo.reward import Reward
from cosmos_reason1.dispatcher.data import CosmosDataset
from torch.utils.data import DataLoader
import cosmos_reason1.dispatcher.command as command
import asyncio
import time
from cosmos_reason1.utils.redis_stream import RedisStreamHandler
from cosmos_reason1.dispatcher.status import (
    PolicyStatus,
    PolicyStatusManager,
    RolloutStatus,
    RolloutStatusManager,
)
from cosmos_reason1.policy.config import Config
import os
import signal
import threading
from tqdm import tqdm


class Controller:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Controller, cls).__new__(cls)
            cls._instance.init_dist()
        return cls._instance

    def __init__(self):
        if not hasattr(self, "policy_replicas"):
            self.init_dist()
        self.init_status()

    def init_status(self):
        self.policy_status_manager = PolicyStatusManager()
        self.rollout_status_manager = RolloutStatusManager()
        self.epoch = 0
        # self.stat_prompt_tokens_count = 0
        # self.stat_completion_tokens_count = 0
        # self.stat_n_samples = 0
        # self.begin_time = None

    def init_dist(self):
        self.policy_replicas: Dict[str, Replica] = {}
        self.rollout_replicas: Dict[str, Replica] = {}
        self.config = None

        self.temp_kv_store = {}

        self.rollout_trajectory = Queue(maxsize=constant.COSMOS_ROLLOUT_TRAJECTORY_SIZE)

        self.life_cycle_lock = asyncio.Lock()

        # Buffer for rollouts in case policy replicas are not ready
        self.rollout_buffer = Queue()
        self.policy_init_done = False
        self.rollout_init_done = False
        self.shut_down_event = threading.Event()

    def init_redis(self, host: str, port: int, wait_retries: int = 60000):
        self.redis_controller = RedisStreamHandler(ip=host, port=port)
        # Wait for redis to be ready
        for _ in range(wait_retries):
            try:
                self.redis_controller.redis_client.ping()
                time.sleep(2.0)
                break
            except Exception:
                pass
        self.redis_controller.redis_client.ping()
        logger.info(f"[Controller] Init redis at {host}:{port}")

    def set_config(self, config: Config):
        self.config = config
        task_type = config.train.train_policy.type
        self.is_cosmos = "cosmos" in config.train.train_policy.dataset_name.lower()

        # Init wandb
        if config.logging.enable_logging:
            if is_wandb_available():
                init_wandb(config)
            else:
                logger.warning(
                    "Wandb is not available. Please install it to use wandb logging features."
                )

        # Pre-process dataset
        if self.is_cosmos:
            util.prepare_cosmos_data(config=config)

        if task_type == "grpo":
            reward_func = Reward(config.train.train_policy.reward_function)
            logger.info(
                f"[Controller] Using reward function {config.train.train_policy.reward_function} for GRPO"
            )
            self.rl_algo = REGISTERED_ALGOs[constant.Algo.GRPO](reward_fn=reward_func)
            try:
                # Load GRPO dataset
                dataset = CosmosDataset(config=config)
                self.dataset = dataset
                self.train_dataloader = DataLoader(
                    self.dataset.train_set, batch_size=1, shuffle=True
                )
                self.total_step = len(self.dataset.train_set) * self.config.train.epoch
                self.pbar = tqdm(total=self.total_step, desc="[Controller] Step")
                self.train_dataloader_iter = iter(self.train_dataloader)
            except Exception as e:
                logger.error(f"[Controller] Fail to load dataset: {e}")
        # from transformers import AutoTokenizer

        # self.tokenizer = AutoTokenizer.from_pretrained(config.policy.model_name_or_path)

    async def get_commands(self, replica_name: str) -> List[command.Command]:
        if (
            replica_name not in self.policy_replicas
            and replica_name not in self.rollout_replicas
        ):
            logger.info(
                f"[Controller] Replica {replica_name} not found in both policy and rollout. Return empty commands"
            )
            return []
        if replica_name in self.policy_replicas:
            # query commands from policy replica
            target_replicas = self.policy_replicas
        else:
            target_replicas = self.rollout_replicas

        replica = target_replicas[replica_name]
        commands = []
        while not replica.command_queue.empty():
            commands.append(replica.command_queue.get_nowait())
        return commands

    async def update_kv_store(self, key: str, value: str):
        self.temp_kv_store[key] = value

    async def get_kv_store(self, key: str) -> str:
        return self.temp_kv_store.get(key)

    """
    Rollout functionality
    """

    async def get_batched_prompt(self, n) -> Tuple[List[Tuple[int, str]], bool]:
        # query n prompts from the dataset
        prompt_id_and_str_list: List[Tuple[int, str]] = []
        is_end = False
        for _ in range(n):
            try:
                idx, prompt = next(self.train_dataloader_iter)
                self.pbar.update(1)
                logger.debug(
                    f"[Controller] Epoch {self.epoch} / {self.config.train.epoch}, get prompt at idx {idx}, prompt {prompt}"
                )
            except StopIteration:
                logger.info(f"[Controller] Epoch {self.epoch} finished.")
                self.epoch += 1
                if self.epoch <= self.config.train.epoch:
                    logger.info(f"[Controller] Epoch {self.epoch} start.")
                    self.train_dataloader_iter = iter(self.train_dataloader)
                    idx, prompt = next(self.train_dataloader_iter)
                    self.pbar.update(1)
                else:
                    logger.info(
                        "[Controller] All epochs finished, start stopping all replicas."
                    )
                    is_end = True
                    break
            idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            prompt_id_and_str_list.append(
                (
                    idx,
                    prompt[0]
                    if isinstance(prompt, list) or isinstance(prompt, tuple)
                    else prompt,
                )
            )
        return prompt_id_and_str_list, is_end

    def query_reference_answer(self, prompt_idx: int) -> str:
        return self.dataset.train_set.query_reference_answer(prompt_idx)

    async def trigger_data_fetch_and_training(self):
        sorted_replicas = [
            replica
            for replica in sorted(self.policy_replicas.values(), key=lambda x: x.name)
            if replica.all_atoms_arrived
        ]
        if len(sorted_replicas) == 0:
            return
        if (
            all(
                [
                    replica.pending_rollouts
                    >= self.config.train.train_batch_per_replica
                    for replica in sorted_replicas
                ]
            )
            and self.policy_status_manager.all_ready_or_reduced()
        ):
            for replica in sorted_replicas:
                """
                Here we need to trigger a new data fetch commands for continuing training
                """
                command.DataFetchCommand.trigger(
                    replica=replica,
                    items_count=self.config.train.train_batch_per_replica,
                    global_step=self.policy_status_manager.completed_train_step() + 1,
                    redis_handler=self.redis_controller,
                )
                self.policy_status_manager.set_status(
                    replica.name, PolicyStatus.RUNNING
                )
                replica.pending_rollouts -= self.config.train.train_batch_per_replica

    async def handle_rollout_end_ack(
        self, extra_info: Dict[str, Any], replica_name: str
    ):
        if "is_end" in extra_info:
            assert extra_info["is_end"] in [True, "True", "true"]
            logger.info(
                f"[Controller] Rollout {extra_info['is_end']} is end, update status."
            )
            self.rollout_status_manager.set_status(replica_name, RolloutStatus.END)
        await self.trigger_replica_end_signal()

    async def trigger_replica_end_signal(self):
        sorted_replicas = [
            replica
            for replica in sorted(self.policy_replicas.values(), key=lambda x: x.name)
            if replica.all_atoms_arrived
        ]
        if len(sorted_replicas) == 0:
            return
        # Check if all replicas are ready and all rollouts are finished
        # and all replicas consume all rollouts up to stop the system
        # Later we may rearrange to make unbalanced rollouts comsumable
        if (
            not all(
                [
                    replica.pending_rollouts
                    >= self.config.train.train_batch_per_replica
                    for replica in sorted_replicas
                ]
            )
            and self.rollout_status_manager.all_end()
            and self.policy_status_manager.all_ready_or_reduced()
        ):
            for replica in sorted_replicas:
                if replica.all_atoms_arrived:
                    command.StopCommand.trigger(
                        replica=replica, redis_handler=self.redis_controller
                    )
                    await replica.put_rollout(
                        Rollout(
                            prompt_idx=-1,
                            prompt="",
                            completion="",
                            extra_info={"is_end": True},
                            reward=0.0,
                            advantage=0.0,
                        ),
                        self.redis_controller,
                    )
                    self.policy_status_manager.set_status(
                        replica.name, PolicyStatus.END
                    )
            for replica in self.rollout_replicas.values():
                if replica.all_atoms_arrived:
                    command.StopCommand.trigger(
                        replica=replica, redis_handler=self.redis_controller
                    )
            self.shut_down_event.set()

    def set_replica_timestamp(self, replica_name: str, timestamp: int):
        if not (
            replica_name in self.policy_replicas
            or replica_name in self.rollout_replicas
        ):
            logger.error(
                f"[Controller] Replica {replica_name} not found in both policy and rollout."
            )
            return
        if replica_name in self.policy_replicas:
            self.policy_status_manager.set_timestamp(replica_name, timestamp)
        else:
            self.rollout_status_manager.set_timestamp(replica_name, timestamp)

    def get_replica_timestamp(self, replica_name: str):
        assert (
            replica_name in self.policy_replicas
            or replica_name in self.rollout_replicas
        )
        if replica_name in self.policy_replicas:
            return self.policy_status_manager.get_timestamp(replica_name)
        else:
            return self.rollout_status_manager.get_timestamp(replica_name)

    async def maintain_replica_life_status(self, now: int):
        """
        Maintain the life status of the policy and rollout replicas.
        now: current timestamp in seconds.
        """
        # iterate the policy and rollout replicas
        dead_policy_replicas = self.policy_status_manager.maintain_life_status(now)
        dead_rollout_replicas = self.rollout_status_manager.maintain_life_status(now)
        for replica_name in dead_policy_replicas:
            logger.warning(
                f"[Controller] Policy replica {replica_name} is lost, unregister it from controller."
            )
            await self.unregister(replica_name)
        for replica_name in dead_rollout_replicas:
            logger.warning(
                f"[Controller] Rollout replica {replica_name} is lost, unregister it from controller."
            )
            await self.unregister(replica_name)

    async def put_rollouts(self, rollouts: List[Rollout]):
        """
        Dispatch the rollouts to the policy replicas in a round-robin manner.
        """
        # if self.begin_time is None:
        #     self.begin_time = time.time()
        for rollout in rollouts:
            # self.stat_prompt_tokens_count += len(self.tokenizer.encode(rollout.prompt))
            # self.stat_completion_tokens_count += len(
            #     self.tokenizer.encode(rollout.completion)
            # )
            # self.stat_n_samples += 1
            await self.put_rollout(rollout)

        # # Print pending rollouts inside all policy replicas
        # pending_count = 0
        # for replica in self.policy_replicas.values():
        #     pending_count += replica.pending_rollouts

        # elapsed_time_in_seconds = time.time() - self.begin_time
        # logger.info(
        #     f"[Controller] Stat: {self.stat_n_samples} samples, {self.stat_prompt_tokens_count} prompt tokens, {self.stat_completion_tokens_count} completion tokens, {pending_count} pending rollouts, {elapsed_time_in_seconds} seconds elapsed"
        # )

    async def put_rollout(self, rollout: Rollout):
        """
        Dispatch the rollout to the policy replicas in a round-robin manner.
        It is that replica's responsibility to dispatch the rollout to further (DP_SHARD) atoms.
        """
        self.rollout_buffer.put(rollout)

        sorted_replicas = [
            replica
            for replica in sorted(self.policy_replicas.values(), key=lambda x: x.name)
            if replica.all_atoms_arrived
        ]
        if len(sorted_replicas) == 0:
            return

        while not self.rollout_buffer.empty():
            rollout = self.rollout_buffer.get()
            history_trajectory = list(self.rollout_trajectory.queue)

            found = False
            if len(history_trajectory) > 0:
                for last_replica_name in reversed(history_trajectory):
                    for i, replica in enumerate(sorted_replicas):
                        if replica.name == last_replica_name:
                            found = True
                            break
                    if found:
                        next_replica_name = sorted_replicas[
                            (i + 1) % len(sorted_replicas)
                        ].name
                        break
            if not found:
                next_replica_name = sorted_replicas[0].name
            util.put_with_overwrite(self.rollout_trajectory, next_replica_name)
            await self.policy_replicas[next_replica_name].put_rollout(
                rollout, self.redis_controller
            )
            await self.trigger_data_fetch_and_training()
            await self.trigger_replica_end_signal()

    """
    State of controller
    """

    def policy_mesh_and_group_size(self) -> tuple[List[str], List[int]]:
        mesh_names = copy.deepcopy(MESH_NAMES)
        group_sizes = []
        for replica in self.policy_replicas.values():
            group_sizes.append(replica.group_size)
            break

        return mesh_names, group_sizes

    def rollout_mesh_and_group_size(self) -> tuple[List[str], List[int]]:
        mesh_names = copy.deepcopy(MESH_NAMES)
        group_sizes = []
        for replica in self.rollout_replicas.values():
            group_sizes.append(replica.group_size)
            break

        return mesh_names, group_sizes

    async def train_ack(self, replica_name: str, iteration_count: int):
        if replica_name not in self.policy_replicas:
            raise Exception(f"Replica {replica_name} not found")

        replica = self.policy_replicas[replica_name]

        self.policy_status_manager.set_status(replica.name, PolicyStatus.BACKWARDED)
        self.policy_status_manager.set_status(replica.name, PolicyStatus.REDUCED)
        if self.policy_status_manager.all_reduced():
            # All replicas have been reduced, trigger allreduce
            optimize_step = self.policy_status_manager.completed_optimize_step()
            need_sync_weight = (
                optimize_step % self.config.train.sync_weight_interval == 0
                and not self.rollout_status_manager.all_end()
            )
            # All replicas have been reduced, trigger weight sync
            if need_sync_weight:
                any_loaded_replica = None
                for replica in self.policy_replicas.values():
                    if not replica.all_atoms_arrived:
                        continue
                    # update the weight version of policy
                    replica.weight_step = optimize_step

                    if any_loaded_replica is None:
                        any_loaded_replica = replica
                    else:
                        self.policy_status_manager.set_status(
                            replica.name, PolicyStatus.READY
                        )
                any_loaded_rollout_replica = None
                valid_rollout_replicas = []
                for rollout_replica in self.rollout_replicas.values():
                    if rollout_replica.all_atoms_arrived:
                        if any_loaded_rollout_replica is None:
                            any_loaded_rollout_replica = rollout_replica
                        valid_rollout_replicas.append(rollout_replica)
                command.PolicyToRolloutUnicastCommand.trigger(
                    src_replica=any_loaded_replica,
                    dst_replica=any_loaded_rollout_replica,
                    src_replica_size=self.policy_atoms_in_replica,
                    dst_replica_size=self.rollout_atoms_in_replica,
                    redis_handler=self.redis_controller,
                    optimize_step=optimize_step,
                    status_manager=self.rollout_status_manager,
                )
                if len(valid_rollout_replicas) > 1:
                    command.RolloutToRolloutBroadcastCommand.trigger(
                        src_replica=any_loaded_rollout_replica,
                        dst_replicas=valid_rollout_replicas,
                        redis_handler=self.redis_controller,
                        optimize_step=optimize_step,
                        status_manager=self.rollout_status_manager,
                    )
                self.policy_status_manager.set_status(
                    any_loaded_replica.name, PolicyStatus.READY
                )
            else:
                # No need to trigger allreduce, just trigger the next round of weight updating
                for replica in self.policy_replicas.values():
                    if replica.all_atoms_arrived:
                        self.policy_status_manager.set_status(
                            replica.name, PolicyStatus.READY
                        )
            await self.trigger_data_fetch_and_training()
            await self.trigger_replica_end_signal()

    """
    Life-cycle of controller
    """

    async def register(self, atom: Atom, role: Role):
        async with self.life_cycle_lock:
            target_cache = (
                self.policy_replicas if role == Role.POLICY else self.rollout_replicas
            )
            replica = target_cache.get(atom.replica_name)
            if replica is None:
                replica = Replica(atom.replica_name, role, [atom])
                target_cache[atom.replica_name] = replica
            else:
                replica.arrive(atom)
            atom.bind_replica(replica)

            # set time stamp for this replica
            self.set_replica_timestamp(atom.replica_name, int(time.time()))

            if self.config is not None and self.config.train.train_policy.type != "sft":
                await self.post_register(atom, role)
            return replica

    async def unregister(self, replica_name: str):
        async with self.life_cycle_lock:
            if replica_name in self.policy_replicas:
                self_replica = self.policy_replicas.pop(replica_name)
                left_valid_replicas = set(
                    replica
                    for replica in self.policy_replicas.values()
                    if replica.all_atoms_arrived
                )
                if self_replica.name in self.policy_status_manager.status:
                    self.policy_status_manager.set_status(
                        self_replica.name, PolicyStatus.DELETED
                    )
            elif replica_name in self.rollout_replicas:
                self_replica = self.rollout_replicas.pop(replica_name)
                left_valid_replicas = set(
                    replica
                    for replica in self.rollout_replicas.values()
                    if replica.all_atoms_arrived
                )
                self.rollout_status_manager.pop(replica_name)
            else:
                raise Exception(f"[Controller] Replica {replica_name} not found")
            if self_replica.in_mesh:
                if len(left_valid_replicas) > 0:
                    # Here we need to trigger a new mesh building command even if there is only one replica left
                    # because the existing mesh is not valid anymore
                    command.BuildMeshCommand.trigger(
                        list(left_valid_replicas), redis_handler=self.redis_controller
                    )
        await self.shut_down()

    async def shut_down(self):
        if (
            len(self.policy_replicas) == 0
            and len(self.rollout_replicas) == 0
            and (
                self.shut_down_event.is_set()
                or self.config.train.train_policy.type == "sft"
            )
        ):
            logger.info("[Controller] Shutting down...")
            os.kill(os.getpid(), signal.SIGINT)

    async def post_register(self, atom: Atom, role: Role):
        if atom.replica.all_atoms_arrived:
            logger.info(
                f"[Controller] All atoms of Replica {atom.replica.name} has been set."
            )
            if role == Role.POLICY:
                self.policy_status_manager.set_status(
                    atom.replica.name, RolloutStatus.UNINITIALIZED
                )
                # Check total valid policy replicas
                valid_replicas = []
                if not hasattr(self, "policy_atoms_in_replica"):
                    self.policy_atoms_in_replica = int(math.prod(atom.group_size))

                for replica in self.policy_replicas.values():
                    if replica.all_atoms_arrived:
                        valid_replicas.append(replica)

                if len(valid_replicas) == 1:
                    assert not hasattr(
                        self, "_first_policy_replica_arrived"
                    ), "Expect only one policy replica to load weight during training process"
                    self._first_policy_replica_arrived = True
                    # This is the first policy replica to arrive, it is responsible for weight initialization
                    command.WeightResumeCommand.trigger(
                        atom.replica, redis_handler=self.redis_controller
                    )

                    for rollout_replica in self.rollout_replicas.values():
                        if rollout_replica.all_atoms_arrived:
                            command.PolicyToRolloutUnicastCommand.trigger(
                                src_replica=atom.replica,
                                dst_replica=rollout_replica,
                                src_replica_size=self.policy_atoms_in_replica,
                                dst_replica_size=self.rollout_atoms_in_replica,
                                redis_handler=self.redis_controller,
                                optimize_step=atom.replica.weight_step,
                                status_manager=self.rollout_status_manager,
                            )
                            self.rollout_status_manager.set_status(
                                rollout_replica.name, RolloutStatus.READY
                            )
                            break
                if (
                    len(valid_replicas)
                    == self.config.policy.parallelism.n_init_replicas
                ):
                    if not self.policy_init_done:
                        self.policy_init_done = True

                    # Trigger mesh building (Typically only occurs during initialization)
                    if self.config.policy.parallelism.n_init_replicas == 1:
                        # Only one policy replica, no need to build mesh
                        logger.info(
                            "Only one policy replica required, no need build mesh."
                        )
                        self.policy_status_manager.set_status(
                            atom.replica.name, PolicyStatus.READY
                        )
                        self.policy_status_manager.set_ranks({atom.replica.name: 0})
                    else:
                        # 1. Trigger mesh building
                        replicas_to_rank = command.BuildMeshCommand.trigger(
                            valid_replicas, redis_handler=self.redis_controller
                        )
                        self.policy_status_manager.set_ranks(replicas_to_rank)

                        # 2. Trigger weight/optimizer state synchronization
                        initialized_replica = None
                        for replica in valid_replicas:
                            if replica.weights_loaded_in_view_of_command:
                                initialized_replica = replica
                                break
                        assert (
                            initialized_replica is not None
                        ), "No replica was selected to load weights"
                        command.PolicyToPolicyBroadcastCommand.trigger(
                            src_replica=initialized_replica,
                            dst_replicas=valid_replicas,
                            redis_handler=self.redis_controller,
                        )
                        for replica in valid_replicas:
                            self.policy_status_manager.set_status(
                                replica.name, PolicyStatus.READY
                            )

                elif (
                    len(valid_replicas) < self.config.policy.parallelism.n_init_replicas
                ):
                    logger.info(
                        f"Waiting for {self.config.policy.parallelism.n_init_replicas - len(valid_replicas)} more replicas to arrive"
                    )
                else:
                    # This occurs when new dynamic scaling is triggered
                    initialized_replica = None
                    for replica in valid_replicas:
                        if replica.weights_loaded_in_view_of_command:
                            initialized_replica = replica
                            break
                    assert (
                        initialized_replica is not None
                    ), "No replica was selected to load weights"
                    replicas_to_rank = command.BuildMeshCommand.trigger(
                        valid_replicas, redis_handler=self.redis_controller
                    )
                    self.policy_status_manager.set_ranks(replicas_to_rank)
                    command.PolicyToPolicyUnicastCommand.trigger(
                        src_replica=initialized_replica,
                        dst_replica=atom.replica,
                        redis_handler=self.redis_controller,
                    )
                    self.policy_status_manager.set_status(
                        atom.replica.name, PolicyStatus.READY
                    )

            elif role == Role.ROLLOUT:
                # set replica weight to uninitialized
                self.rollout_status_manager.set_status(
                    atom.replica.name, RolloutStatus.UNINITIALIZED
                )
                # Check total valid rollout replicas
                valid_replicas = []
                if not hasattr(self, "rollout_atoms_in_replica"):
                    self.rollout_atoms_in_replica = int(math.prod(atom.group_size))
                for replica in self.rollout_replicas.values():
                    if replica.all_atoms_arrived:
                        valid_replicas.append(replica)

                if len(valid_replicas) == 1:
                    any_loaded_policy_replica = None
                    for replica in self.policy_replicas.values():
                        if replica.weights_loaded_in_view_of_command:
                            any_loaded_policy_replica = replica
                            break
                    if any_loaded_policy_replica is not None:
                        command.PolicyToRolloutUnicastCommand.trigger(
                            src_replica=any_loaded_policy_replica,
                            dst_replica=atom.replica,
                            src_replica_size=self.policy_atoms_in_replica,
                            dst_replica_size=self.rollout_atoms_in_replica,
                            redis_handler=self.redis_controller,
                            optimize_step=any_loaded_policy_replica.weight_step,
                            status_manager=self.rollout_status_manager,
                        )
                        self.rollout_status_manager.set_status(
                            atom.replica.name, RolloutStatus.READY
                        )
                    else:
                        logger.info(
                            "No policy replica was loaded, will be rescheduled after first policy replica is loaded"
                        )
                if (
                    len(valid_replicas)
                    >= self.config.rollout.parallelism.n_init_replicas
                ):
                    # Rollout replicas all arrived, set the timestamp for them as the initialization timestamp
                    # For hearbeat check.
                    for replica in valid_replicas:
                        self.set_replica_timestamp(replica.name, int(time.time()))

                    was_already_initialized = self.rollout_init_done
                    self.rollout_init_done = True
                    if len(valid_replicas) > 1:
                        # Trigger Mesh building
                        command.BuildMeshCommand.trigger(
                            valid_replicas, redis_handler=self.redis_controller
                        )
                        if not was_already_initialized:
                            # Trigger RolloutToRolloutBroadcastCommand only once after all initial rollout replicas are loaded
                            any_loaded_rollout_replica = None
                            for replica in valid_replicas:
                                if replica.weights_loaded_in_view_of_command:
                                    any_loaded_rollout_replica = replica
                                    break
                            assert (
                                any_loaded_rollout_replica is not None
                            ), "[Controller] No rollout replica was selected to broadcast weights"
                            command.RolloutToRolloutBroadcastCommand.trigger(
                                src_replica=any_loaded_rollout_replica,
                                dst_replicas=valid_replicas,
                                redis_handler=self.redis_controller,
                                optimize_step=any_loaded_rollout_replica.weight_step,
                                status_manager=self.rollout_status_manager,
                            )
                            for replica in valid_replicas:
                                if any_loaded_rollout_replica.name != replica.name:
                                    self.rollout_status_manager.set_status(
                                        replica.name, RolloutStatus.READY
                                    )
                        else:
                            # The new rollout replicas will be broadcasted in the next round of weight broadcast
                            pass
                else:
                    logger.info(
                        f"Waiting for {self.config.rollout.parallelism.n_init_replicas - len(valid_replicas)} more replicas to arrive"
                    )
            else:
                raise Exception(f"[Controller] Unknown role during register: {role}")
