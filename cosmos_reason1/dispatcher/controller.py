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

from typing import List, Dict, Tuple, Any, Optional, Callable
import copy
import math
import torch
import subprocess
import atexit
import sys
import uuid
import asyncio
import time
import itertools
import os
import signal
import threading
import numpy as np
from queue import Queue
from cosmos_reason1.dispatcher.replica import Atom, Replica, Rollout, RolloutGroup
from cosmos_reason1.dispatcher.protocol import Role, MESH_NAMES
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.wandb_logger import (
    is_wandb_available,
    init_wandb,
    log_wandb,
)
import cosmos_reason1.utils.util as util
import cosmos_reason1.utils.network_util as network_util
import cosmos_reason1.utils.constant as constant
from cosmos_reason1.dispatcher.algo.base import REGISTERED_ALGOs
from cosmos_reason1.dispatcher.algo.reward import Reward
from cosmos_reason1.dispatcher.data import (
    CosmosDataset,
    RLPayload,
    CosmosValidationDataset,
)
from torch.utils.data import DataLoader, Dataset
import cosmos_reason1.dispatcher.command as command
from cosmos_reason1.dispatcher.command import ProcessPhase
from cosmos_reason1.utils.redis_stream import RedisStreamHandler
from cosmos_reason1.dispatcher.status import (
    PolicyStatus,
    PolicyStatusManager,
    RolloutStatus,
    RolloutStatusManager,
)
from cosmos_reason1.utils.checkpoint import CheckpointMananger
from cosmos_reason1.policy.config import Config, SubProfilerConfig
from cosmos_reason1.dispatcher.protocol import SetProfileRequest
from transformers import AutoTokenizer
import tempfile
from cosmos_reason1.dispatcher.data.packer.base import DataPacker
from tqdm import tqdm


class Controller:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Controller, cls).__new__(cls)
            cls._instance._init_dist()
        return cls._instance

    def __init__(self):
        if not hasattr(self, "policy_replicas"):
            self._init_dist()
        self._init_status()

    def _init_status(self):
        self.policy_status_manager = PolicyStatusManager()
        self.rollout_status_manager = RolloutStatusManager()
        self.epoch = 1
        self.controller_step = 0
        self.stat_prompt_tokens_count = 0
        self.stat_completion_tokens_count = 0
        self.stat_n_samples = 0
        self.begin_time = None
        # nccl error check
        self.post_ncclerror_policy_invoke_id = 0
        self.post_ncclerror_rollout_invoke_id = 0
        self.phase = ProcessPhase.TRAIN

    def _init_dist(self):
        self.policy_replicas: Dict[str, Replica] = {}
        self.rollout_replicas: Dict[str, Replica] = {}
        self.config = None

        self.temp_kv_store = {}

        self.rollout_trajectory = Queue(maxsize=constant.COSMOS_ROLLOUT_TRAJECTORY_SIZE)

        self.life_cycle_lock = asyncio.Lock()

        # Buffer for rollouts in case policy replicas are not ready
        self.rollout_buffer = Queue()
        self.per_step_rollout_buffer = Queue()
        self.policy_init_done = False
        self.rollout_init_done = False
        self.shut_down_event = threading.Event()

    def setup(
        self,
        config: Config,
        redis_port: int,
        redis_logfile_path: str,
        dataset: Optional[Dataset] = None,
        reward_fns: Optional[List[Callable]] = None,
        data_packer: Optional[DataPacker] = None,
        val_dataset: Optional[Dataset] = None,
        val_reward_fns: Optional[List[Callable]] = None,
        val_data_packer: Optional[DataPacker] = None,
    ):
        if self.config is not None:
            raise Exception(
                "[Controller] Config has been set. Please do not call setup again."
            )

        self.config = config
        task_type = config.train.train_policy.type
        self.tokenizer = util.retry(AutoTokenizer.from_pretrained)(
            config.policy.model_name_or_path
        )

        if "wandb" in config.logging.logger and is_wandb_available():
            self.wandb_run = init_wandb(config)
        else:
            logger.warning(
                "Wandb is not available. Please install it to use wandb logging features."
            )

        self.is_rl = task_type != "sft"
        self.sft_user_dataset = dataset if not self.is_rl else None
        self.user_data_packer = data_packer
        self.user_val_data_packer = val_data_packer
        if self.is_rl:
            if dataset is not None:
                # Do simple sanity check
                assert isinstance(dataset, Dataset)
                self.dataset = CosmosDataset(
                    config=config, train_set=dataset, tokenizer=self.tokenizer
                )
                logger.info(
                    "[Controller] Using provided dataset for training, dataset specification in the toml config will be ignored"
                )
            else:
                self.dataset = CosmosDataset(config=config, tokenizer=self.tokenizer)
            self.rl_algo = REGISTERED_ALGOs[constant.Algo.GRPO](
                reward_fn=Reward(
                    config=config,
                    tokenier=self.tokenizer,
                    reward_function=config.train.train_policy.reward_function,
                    explicit_reward_fn=reward_fns,
                ),
                unbiased=config.train.train_policy.unbiased_advantage,
            )
            self.train_dataloader = DataLoader(
                self.dataset.train_set,
                batch_size=1,  # batch size is 1 is mandatory
                shuffle=config.train.train_policy.dataloader_shuffle,
                num_workers=config.train.train_policy.dataloader_num_workers,
                prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
                collate_fn=RLPayload.collate_fn,
            )
            self.train_dataloader_iter = iter(self.train_dataloader)

            self.val_rollouts = []
            if config.train.enable_validation:
                # Only when dataset is None, the config of train dataset is valid and can be used for validation dataset.
                if not config.validation.dataset.name:
                    config.validation.dataset.name = (
                        config.train.train_policy.dataset.name
                    )
                    config.validation.dataset.subset = (
                        config.train.train_policy.dataset.subset
                    )
                    config.validation.dataset.revision = (
                        config.train.train_policy.dataset.revision
                    )
                if not config.validation.dataset.test_split:
                    assert (
                        config.train.train_policy.dataset.name
                        == config.validation.dataset.name
                        and config.train.train_policy.dataset.subset
                        == config.validation.dataset.subset
                    ), "Validation dataset must have the same dataset name and subset as training dataset if validation test_split needs use training dataset split."
                    config.validation.dataset.test_split = (
                        config.train.train_policy.dataset.train_split
                    )
                    config.validation.dataset.test_size = (
                        config.train.train_policy.dataset.test_size
                    )

                if val_dataset is not None:
                    # Do simple sanity check
                    assert isinstance(val_dataset, Dataset)
                    self.val_dataset = CosmosValidationDataset(
                        config=config, val_set=val_dataset, tokenizer=self.tokenizer
                    )
                    logger.info(
                        "[Controller] Using provided validation dataset for validation, dataset specification in the toml config will be ignored"
                    )
                else:
                    self.val_dataset = CosmosValidationDataset(
                        config=config, tokenizer=self.tokenizer
                    )
                if not config.validation.reward_function:
                    if val_reward_fns is None:
                        val_reward_fns = reward_fns
                        if val_reward_fns is not None:
                            logger.warning(
                                "[Controller] No validation reward functions provided, using the same reward functions as training."
                            )
                    config.validation.reward_function = (
                        config.train.train_policy.reward_function
                    )
                    logger.warning(
                        "[Controller] No validation reward function config specified, using the same reward function as training."
                    )
                self.val_rl_algo = REGISTERED_ALGOs[constant.Algo.GRPO](
                    reward_fn=Reward(
                        config=config,
                        tokenier=self.tokenizer,
                        reward_function=config.validation.reward_function,
                        explicit_reward_fn=val_reward_fns,
                    )
                )

                self.val_dataloader = DataLoader(
                    self.val_dataset.val_set,
                    batch_size=1,  # batch size is 1 is mandatory
                    shuffle=config.train.train_policy.dataloader_shuffle,
                    num_workers=config.train.train_policy.dataloader_num_workers,
                    prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
                    collate_fn=RLPayload.collate_fn,
                )
                self.val_dataloader_iter = iter(self.val_dataloader)
            else:
                self.val_dataset = None
                self.val_dataloader = None
                self.val_dataloader_iter = None
                self.val_rl_algo = None

            self.policy_status_manager.set_train_batch_per_replica(
                config.train.train_batch_per_replica
            )
            num_data_samples = (
                len(self.dataset.train_set)
                * config.rollout.n_generation
                * self.config.train.epoch
            )
            # Resume from checkpoint if needed
            if config.train.resume:
                self.ckpt_manager = CheckpointMananger(config)
                command.PolicyToRolloutUnicastCommand.do_weight_sync_check_flag = False
                try:
                    ckpt_path = self.ckpt_manager.get_ckpt_path()
                    ckpt_extra_vars = self.ckpt_manager.load_extra_info(
                        os.path.join(ckpt_path, "extra_info_rank_0.pth")
                    )
                    ckpt_start_step = ckpt_extra_vars.get("step", 0)
                    num_data_samples = ckpt_extra_vars["remain_samples_num"]
                    self.policy_status_manager.set_start_step(ckpt_start_step)
                    self.epoch = (
                        config.train.epoch
                        - (
                            math.ceil(
                                num_data_samples
                                / (
                                    len(self.dataset.train_set)
                                    * config.rollout.n_generation
                                )
                            )
                        )
                        + 1
                    )
                    if config.train.train_policy.dataloader_shuffle:
                        logger.warning(
                            "Since dataloader_shuffle is True, the dataloader status cannot be resumed identically."
                        )
                    data_loader_bias = (
                        num_data_samples // config.rollout.n_generation
                    ) % len(self.dataset.train_set)
                    for _ in range(data_loader_bias):
                        try:
                            next(self.train_dataloader_iter)
                        except StopIteration:
                            logger.warning(
                                "[Controller] Data loader bias adjustment: reached end of dataset."
                            )
                            self.train_dataloader_iter = iter(self.train_dataloader)
                except Exception as e:
                    logger.error(f"Failed to resume from checkpoint: {e}")
            self.policy_status_manager.set_num_data_samples(num_data_samples)
        else:
            self.rl_algo = None

        redis_free_port = util.find_available_port(redis_port)
        self.config.redis = str(redis_free_port)

        ips = network_util.get_eth_ips()
        if len(ips) > 0:
            self.config.eth_ips = ";".join(ips)

        random_db_file_name = f"cosmos_reason1_{str(uuid.uuid4())}.rdb"
        config_file_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".redis_config.conf"
        )
        redis_cfg_path = util.write_redis_config(
            redis_free_port, redis_logfile_path, file_path=config_file_path.name
        )
        redis_server_cmd = f'redis-server {redis_cfg_path} --dbfilename {random_db_file_name} --save ""'

        redis_server_proc = subprocess.Popen(
            redis_server_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr
        )

        # Check if the redis server started successfully
        redis_server_proc.wait()
        ret_code = redis_server_proc.returncode

        if ret_code is not None and ret_code != 0:
            raise RuntimeError(
                f"Failed to start redis server with command: {redis_server_cmd} with return code {ret_code}"
            )
        else:
            logger.info(
                f"[Controller] Redis server started on port {redis_free_port} with command {redis_server_cmd}"
            )

        self.redis_controller = RedisStreamHandler(
            ips=["0.0.0.0"], port=redis_free_port
        )

        # Register the exit function to be called when the program exits
        def exit_server(redis_server_proc, redis_free_port):
            logger.info("Stopping redis server")
            redis_server_proc.terminate()
            redis_server_proc.wait()

            redis_terminate_cmd = f"redis-cli -p {redis_free_port} shutdown nosave"
            redis_terminate = subprocess.Popen(
                redis_terminate_cmd,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            redis_terminate.wait()
            try:
                os.unlink(config_file_path.name)
            except Exception:
                # best effort to remove the config file
                pass
            logger.info("Redis server stopped.")

        atexit.register(exit_server, redis_server_proc, redis_free_port)

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
        prompt_id_and_payload_list: List[Tuple[int, str]] = []
        is_end = False

        if self.phase != ProcessPhase.TRAIN:
            return (
                prompt_id_and_payload_list,
                is_end,
            )

        # Throttle the generation speed:
        # 1. Detect the current left pending rollouts in all policy replicas.
        # 2. Check the config.train.train_policy.allowed_outdated_steps.
        # 3. If the current pending rollouts is larger than the allowed outdated version count, reduce the number of prompts to generate.
        current_pending_rollouts = sum(
            replica.pending_rollouts for replica in self.policy_replicas.values()
        )
        if (
            current_pending_rollouts
            > self.config.train.train_policy.allowed_outdated_steps
            * len(self.policy_replicas)
            * self.config.train.train_batch_per_replica
        ):
            logger.warning(
                f"[Controller] Current pending rollouts {current_pending_rollouts} is larger than the allowed outdated version count {self.config.train.train_policy.allowed_outdated_steps * len(self.policy_replicas)}."
            )
            n = 1

        for _ in range(n):
            payload = None
            try:
                idx, payload = next(self.train_dataloader_iter)
                assert len(idx) == 1
                assert len(payload) == 1
                idx = idx[0]
                payload = payload[0].payload

                self.controller_step += 1
                logger.debug(
                    f"[Controller] Epoch {self.epoch} / {self.config.train.epoch}, Step {self.controller_step}, get prompt at idx {idx}, payload {payload}"
                )
            except StopIteration:
                logger.info(f"[Controller] Epoch {self.epoch} finished.")
                self.epoch += 1
                if self.epoch <= self.config.train.epoch:
                    logger.info(f"[Controller] Epoch {self.epoch} start.")
                    self.train_dataloader_iter = iter(self.train_dataloader)
                    idx, payload = next(self.train_dataloader_iter)
                    assert len(idx) == 1
                    assert len(payload) == 1
                    idx = idx[0]
                    payload = payload[0].payload
                else:
                    logger.info(
                        "[Controller] All epochs finished, start stopping all replicas."
                    )
                    is_end = True
                    break
            idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            prompt_id_and_payload_list.append((idx, payload))

        return prompt_id_and_payload_list, is_end

    async def get_batched_validation_prompt(
        self, n
    ) -> Tuple[List[Tuple[int, str]], bool]:
        assert (
            self.val_dataloader is not None
        ), "Validation dataloader is not initialized."
        assert (
            self.phase == ProcessPhase.VALIDATE
        ), "Controller is not in validation phase."
        prompt_id_and_payload_list: List[Tuple[int, str]] = []
        is_end = False
        for _ in range(n):
            payload = None
            try:
                idx, payload = next(self.val_dataloader_iter)
                assert len(idx) == 1
                assert len(payload) == 1
                idx = idx[0]
                payload = payload[0].payload
            except StopIteration:
                is_end = True
                break
            idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            prompt_id_and_payload_list.append((idx, payload))
        return prompt_id_and_payload_list, is_end

    def query_reference_answer(self, prompt_idx: int) -> Any:
        return self.dataset.train_set.get_reference_answer(prompt_idx)

    async def set_profile(self, request: SetProfileRequest):
        replica_name = request.replica_name

        if replica_name not in self.policy_replicas:
            logger.warning(
                f"[Controller] Replica {replica_name} not found in policy replicas. The profile request takes no effect."
            )
            return {
                "message": "Replica not found in policy replicas. The profile request takes no effect."
            }
        if self.policy_replicas[replica_name].sub_profiler_config.do_profile:
            logger.warning(
                f"[Controller] Replica {replica_name} is already in profile mode. The profile request takes no effect."
            )
            return {
                "message": "Replica is already in profile mode. The profile request takes no effect."
            }
        else:
            kwargs_dict = request.model_dump()
            # remove the replica_name from the kwargs_dict
            kwargs_dict.pop("replica_name")
            # add do_profile to the kwargs_dict
            kwargs_dict["do_profile"] = True
            self.policy_replicas[replica_name].sub_profiler_config = SubProfilerConfig(
                **kwargs_dict
            )
            logger.info(f"[Controller] Set profile mode for replica {replica_name}.")
            return {"message": f"Set replica {replica_name} to profile mode."}

    async def set_trace_path(
        self, replica_name: str, trace_path: str, global_rank: int
    ):
        if replica_name not in self.policy_replicas:
            logger.warning(
                f"[Controller] Replica {replica_name} not found in policy replicas. The trace path request takes no effect."
            )
            return None
        return await self.policy_replicas[replica_name].set_trace_path(
            trace_path, global_rank
        )

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
            remain_samples_num = (
                self.policy_status_manager.get_num_data_samples()
                - self.config.train.train_batch_per_replica * len(sorted_replicas)
            )
            cur_train_step = self.policy_status_manager.completed_train_step() + 1
            total_steps = self.policy_status_manager.get_total_steps()
            for replica in sorted_replicas:
                """
                Here we need to trigger a new data fetch commands for continuing training
                """
                command.DataFetchCommand.trigger(
                    replica=replica,
                    items_count=self.config.train.train_batch_per_replica,
                    global_step=cur_train_step,
                    total_steps=total_steps,
                    remain_samples_num=remain_samples_num,
                    redis_handler=self.redis_controller,
                )
                self.policy_status_manager.set_status(
                    replica.name, PolicyStatus.RUNNING
                )
                replica.pending_rollouts -= self.config.train.train_batch_per_replica

            if self.config.logging.logger:
                self.rollout_statistic(sorted_replicas, cur_train_step, total_steps)

    async def handle_rollout_end_ack(
        self, extra_info: Dict[str, Any], replica_name: str
    ):
        if "is_end" in extra_info:
            assert extra_info["is_end"] in [True, "True", "true"]
            logger.info(f"[Controller] Rollout {replica_name} is end, update status.")
            self.rollout_status_manager.set_status(replica_name, RolloutStatus.END)
        await self.trigger_replica_end_signal()

    async def handle_validation_end_per_replica(self, replica_name: str):
        """
        Handle the end of validation for a specific replica.
        This is called when a validation rollout is finished.
        """
        if self.phase == ProcessPhase.VALIDATE:
            self.rollout_status_manager.set_status(replica_name, RolloutStatus.READY)
            if not self.rollout_status_manager.any_validate():
                logger.debug(
                    "[Controller] All validation rollouts are done, stop validation."
                )
                await self.validate_rollouts()
                self.phase = ProcessPhase.TRAIN
                await self.trigger_data_fetch_and_training()
                await self.trigger_replica_end_signal()

    async def handle_validation_rollout_end_ack(
        self, extra_info: Dict[str, Any], replica_name: str
    ):
        if "is_end" in extra_info:
            assert extra_info["is_end"] in [True, "True", "true"]
            logger.debug(
                f"[Controller] Validation rollout {replica_name} is end, stop validation."
            )
            assert (
                self.phase == ProcessPhase.VALIDATE
            ), "Controller is not in validation phase."
            assert (
                self.rollout_status_manager.get_status(replica_name)
                in [RolloutStatus.VALIDATE, RolloutStatus.END_VALIDATE]
            ), f"Validation rollout {replica_name} is not in validate status, current status: {self.rollout_status_manager.get_status(replica_name)}"
            await self.handle_validation_end_per_replica(replica_name)

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
                            payload="",
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
            logger.warning(
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

    async def put_validation_rollouts(
        self, rollout_groups: List[RolloutGroup], replica_name: str
    ):
        rollouts_list: List[List[Rollout]] = [
            rollout_group.compute_rollouts(self.val_rl_algo)
            for rollout_group in rollout_groups
        ]
        self.val_rollouts.extend(rollouts_list)
        self.val_bar.update(sum(len(rollouts) for rollouts in rollouts_list))

    async def validate_rollouts(self):
        """
        Validate the rollouts from the rollout replicas.
        This is called when the validation phase is triggered.
        """
        assert (
            self.phase == ProcessPhase.VALIDATE
        ), "Controller is not in validation phase."
        rewards = []
        for rollouts in self.val_rollouts:
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
            "val/step": self.policy_status_manager.completed_optimize_step(),
        }
        logger.info(
            f"[Controller] Validation finished, average reward: {avg_reward}, total rollouts: {len(rewards)}, max reward: {max_reward}, min reward: {min_reward}, std reward: {std_reward} at step {self.policy_status_manager.completed_optimize_step()}"
        )
        if "wandb" in self.config.logging.logger and is_wandb_available():
            log_wandb(
                self.wandb_run,
                data=report_data,
                step=self.policy_status_manager.completed_optimize_step(),
            )
        self.val_rollouts.clear()
        self.val_bar.close()

    async def put_rollouts(
        self, valid_rollouts: List[Rollout], invalid_rollouts: List[Rollout]
    ):
        """
        Dispatch the rollouts to the policy replicas in a round-robin manner.
        valid_rollouts: List[Rollout]: The rollouts that have valid rewards
        invalid_rollouts: List[Rollout]: The rollouts that have invalid rewards (all rewards are the same)
        """
        if self.config.train.train_policy.variant == "dapo":
            for rollout in valid_rollouts:
                await self.put_rollout(rollout)
        else:
            for rollout in itertools.chain(valid_rollouts, invalid_rollouts):
                await self.put_rollout(rollout)

        # Statistic
        if self.begin_time is None:
            self.begin_time = time.time()
        for rollout in itertools.chain(valid_rollouts, invalid_rollouts):
            self.stat_completion_tokens_count += len(
                self.tokenizer.encode(rollout.completion)
            )
            self.stat_n_samples += 1

        # Print pending rollouts inside all policy replicas
        pending_count = 0
        for replica in self.policy_replicas.values():
            pending_count += replica.pending_rollouts

        elapsed_time_in_seconds = time.time() - self.begin_time
        logger.info(
            f"[Controller] Stat: {self.stat_n_samples} samples, {self.stat_completion_tokens_count} completion tokens, {pending_count} pending rollouts, {elapsed_time_in_seconds} seconds elapsed"
        )

    async def put_rollout(self, rollout: Rollout):
        """
        Dispatch the rollout to the policy replicas in a round-robin manner.
        It is that replica's responsibility to dispatch the rollout to further (DP_SHARD) atoms.
        """
        if self.config.rollout.include_stop_str_in_output:
            if self.tokenizer.eos_token is not None and rollout.completion is not None:
                if not rollout.completion.endswith(self.tokenizer.eos_token):
                    rollout.completion = rollout.completion + self.tokenizer.eos_token
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
            self.per_step_rollout_buffer.put(rollout)
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
            if self.phase == ProcessPhase.VALIDATE:
                # If we are in validation phase, we need to wait for the validation to finish
                return
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

    def get_start_time_sorted_policies(self) -> List[Replica]:
        """
        Get the policy replicas sorted by their start time.
        """
        return sorted(self.policy_replicas.values(), key=lambda x: x.start_time)

    def get_start_time_sorted_rollouts(self) -> List[Replica]:
        """
        Get the rollout replicas sorted by their start time.
        """
        return sorted(self.rollout_replicas.values(), key=lambda x: x.start_time)

    async def update_policies_initialize(
        self, valid_replicas: List[Replica], target_replica: Replica
    ):
        if not any(
            [replica.weights_loaded_in_view_of_command for replica in valid_replicas]
        ):
            return
        if (
            not self.policy_init_done
            and len(valid_replicas) >= self.config.policy.parallelism.n_init_replicas
        ):
            self.policy_init_done = True
            # Trigger mesh building (Typically only occurs during initialization)

            # we need buildmesh, event there is only one replica. (trigger HANccl buildmesh)
            # 1. Trigger mesh building
            replicas_to_rank = command.BuildMeshCommand.trigger(
                valid_replicas, redis_handler=self.redis_controller
            )
            self.policy_status_manager.set_ranks(replicas_to_rank)

            # 2. Trigger weight/optimizer state synchronization
            if len(valid_replicas) > 1:
                # Only broadcast when there are multiple policy replicas
                initialized_replica = None
                for replica in self.get_start_time_sorted_policies():
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
                    redis_handler=self.redis_controller,
                )
            for replica in valid_replicas:
                self.policy_status_manager.set_status(replica.name, PolicyStatus.READY)
        elif len(valid_replicas) < self.config.policy.parallelism.n_init_replicas:
            logger.info(
                f"Waiting for {self.config.policy.parallelism.n_init_replicas - len(valid_replicas)} more replicas to arrive"
            )
        else:
            if target_replica.name in self.policy_status_manager.policy_to_rank:
                # This replica is already in the mesh, no need to build mesh
                return
            # This occurs when new dynamic scaling is triggered
            initialized_replica = None
            for replica in self.get_start_time_sorted_policies():
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
            replicas_to_rank = command.BuildMeshCommand.trigger(
                valid_replicas, redis_handler=self.redis_controller
            )
            self.policy_status_manager.set_ranks(replicas_to_rank)
            command.PolicyToPolicyUnicastCommand.trigger(
                src_replica=initialized_replica,
                dst_replica=target_replica,
                redis_handler=self.redis_controller,
            )
            self.policy_status_manager.set_status(
                target_replica.name, PolicyStatus.READY
            )

    async def update_rollouts_initialize(
        self, valid_replicas: List[Replica], target_replica: Replica
    ):
        assert target_replica in valid_replicas
        any_loaded_policy_replica = None
        for replica in self.get_start_time_sorted_policies():
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
        if all(
            [
                not replica.weights_loaded_in_view_of_command
                for replica in valid_replicas
            ]
        ):
            # We will tell rollout replica to check the weight sync correctness
            # For the first p2r. Check details in the `trigger`.
            command.PolicyToRolloutUnicastCommand.trigger(
                src_replica=any_loaded_policy_replica,
                dst_replica=target_replica,
                src_replica_size=self.policy_atoms_in_replica,
                dst_replica_size=self.rollout_atoms_in_replica,
                redis_handler=self.redis_controller,
                optimize_step=any_loaded_policy_replica.weight_step,
                status_manager=self.rollout_status_manager,
            )
            self.rollout_status_manager.set_status(
                target_replica.name, RolloutStatus.READY
            )
        if len(valid_replicas) >= self.config.rollout.parallelism.n_init_replicas:
            was_already_initialized = self.rollout_init_done
            self.rollout_init_done = True
            if len(valid_replicas) > 1:
                # Trigger Mesh building
                ranks = command.BuildMeshCommand.trigger(
                    valid_replicas, redis_handler=self.redis_controller
                )
                self.rollout_status_manager.set_ranks(ranks)
                if not was_already_initialized:
                    # Trigger RolloutToRolloutBroadcastCommand only once after all initial rollout replicas are loaded
                    any_loaded_rollout_replica = None
                    for replica in self.get_start_time_sorted_rollouts():
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
                # Only one rollout replica, no need to build mesh
                self.rollout_status_manager.set_ranks({target_replica.name: 0})
        else:
            logger.info(
                f"Waiting for {self.config.rollout.parallelism.n_init_replicas - len(valid_replicas)} more replicas to arrive"
            )

    async def weight_ready(self, replica_name: str):
        if replica_name not in self.policy_replicas:
            raise Exception(f"Replica {replica_name} not found")
        self.policy_replicas[replica_name].weights_loaded_in_view_of_command = True
        self.policy_status_manager.set_status(replica_name, PolicyStatus.READY)
        initialized_replica = None
        for replica in self.policy_replicas.values():
            if replica.all_atoms_arrived and replica.weights_loaded_in_view_of_command:
                # This replica is responsible for weight initialization
                initialized_replica = replica
                break
        assert (
            initialized_replica.name == replica_name
        ), "The replica that is responsible for weight initialization is not the same as the one that sent the weight ready command"
        valid_rollout_replicas = [
            replica
            for replica in self.get_start_time_sorted_rollouts()
            if replica.all_atoms_arrived
        ]
        if len(valid_rollout_replicas) > 0:
            await self.update_rollouts_initialize(
                valid_rollout_replicas, valid_rollout_replicas[0]
            )
        await self.update_policies_initialize(
            [
                replica
                for replica in self.policy_replicas.values()
                if replica.all_atoms_arrived
            ],
            self.policy_replicas[replica_name],
        )

    async def update_rollouts_weights(
        self, policy_replica: Replica, optimize_step: int
    ):
        any_loaded_rollout_replica = None
        valid_rollout_replicas = []
        for rollout_replica in self.get_start_time_sorted_rollouts():
            if rollout_replica.all_atoms_arrived:
                if any_loaded_rollout_replica is None:
                    any_loaded_rollout_replica = rollout_replica
                valid_rollout_replicas.append(rollout_replica)
        if any_loaded_rollout_replica is None:
            return
        command.PolicyToRolloutUnicastCommand.trigger(
            src_replica=policy_replica,
            dst_replica=any_loaded_rollout_replica,
            src_replica_size=self.policy_atoms_in_replica,
            dst_replica_size=self.rollout_atoms_in_replica,
            redis_handler=self.redis_controller,
            optimize_step=policy_replica.weight_step,
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

    async def trigger_validate(self):
        train_step = self.policy_status_manager.completed_optimize_step()
        if (
            self.config.train.enable_validation
            and train_step % self.config.train.validation_freq == 0
        ):
            logger.info(f"[Controller] Trigger validation at train step {train_step}.")
            self.val_bar = tqdm(
                desc="validation",
                total=len(self.val_dataset.val_set)
                * self.config.validation.n_generation,
            )
            self.val_dataloader_iter = iter(self.val_dataloader)
            for replica in self.rollout_replicas.values():
                if replica.all_atoms_arrived and replica.weight_step == train_step:
                    command.ValidateCommand.trigger(
                        replica=replica, redis_handler=self.redis_controller
                    )
                    self.rollout_status_manager.set_status(
                        replica.name, RolloutStatus.VALIDATE
                    )
            self.phase = ProcessPhase.VALIDATE

    async def train_ack(
        self, replica_name: str, iteration_count: int, profile_finished: bool
    ):
        if replica_name not in self.policy_replicas:
            raise Exception(f"Replica {replica_name} not found")

        self.policy_status_manager.set_status(replica_name, PolicyStatus.BACKWARDED)
        self.policy_status_manager.set_status(replica_name, PolicyStatus.REDUCED)
        if self.policy_status_manager.all_reduced():
            # All replicas have been reduced, trigger allreduce
            optimize_step = self.policy_status_manager.completed_optimize_step()
            need_sync_weight = (
                optimize_step % self.config.train.sync_weight_interval == 0
                and not self.rollout_status_manager.all_end()
            ) or (
                self.config.train.enable_validation
                and self.policy_status_manager.completed_optimize_step()
                % self.config.train.validation_freq
                == 0
            )

            if profile_finished:
                # Only reset the do_profile flag if the profile is finished
                logger.debug(f"[Controller] Unset the profile mode of {replica_name}")
                self.policy_replicas[
                    replica_name
                ].sub_profiler_config.do_profile = False

            # All replicas have been reduced, trigger weight sync
            if need_sync_weight:
                any_loaded_replica = None
                for replica in self.get_start_time_sorted_policies():
                    if not replica.all_atoms_arrived:
                        continue
                    # update the weight version of policy
                    replica.weight_step = optimize_step

                    if any_loaded_replica is None:
                        any_loaded_replica = replica
                    self.policy_status_manager.set_status(
                        replica.name, PolicyStatus.READY
                    )
                await self.update_rollouts_weights(any_loaded_replica, optimize_step)
            else:
                # No need to trigger allreduce, just trigger the next round of weight updating
                for replica in self.policy_replicas.values():
                    if replica.all_atoms_arrived:
                        self.policy_status_manager.set_status(
                            replica.name, PolicyStatus.READY
                        )
            await self.trigger_validate()
            if self.phase == ProcessPhase.VALIDATE:
                # If we are in validation phase, we need to wait for the validation to finish
                logger.debug(
                    f"[Controller] Waiting for validation to finish at step {self.policy_status_manager.completed_optimize_step()} before triggering next round of training."
                )
                return
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
            manager = None
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
                manager = self.policy_status_manager
            elif replica_name in self.rollout_replicas:
                self_replica = self.rollout_replicas.pop(replica_name)
                left_valid_replicas = set(
                    replica
                    for replica in self.rollout_replicas.values()
                    if replica.all_atoms_arrived
                )
                await self.handle_validation_end_per_replica(replica_name)
                self.rollout_status_manager.pop(replica_name)
                manager = self.rollout_status_manager
            else:
                raise Exception(f"[Controller] Replica {replica_name} not found")
            if self_replica.in_mesh:
                if len(left_valid_replicas) > 0:
                    # Here we need to trigger a new mesh building command even if there is only one replica left
                    # because the existing mesh is not valid anymore
                    ranks = command.BuildMeshCommand.trigger(
                        list(left_valid_replicas), redis_handler=self.redis_controller
                    )
                    manager.set_ranks(ranks)
                else:
                    manager.set_ranks({})
            manager.remove_from_ranks(replica_name)
        await self.shut_down()

    async def shut_down(self):
        # COSMOS_AT_UNREGISTER_ALL: This environment variable is used to control the shutdown behavior of the controller.
        # If it is set to "EXIT", the controller will exit after unregistering all replicas.
        # If it is set to "WAIT", the controller will continue to run and wait for new replicas to register.
        # If it is not set, the default behavior is to exit.
        mode = os.getenv("COSMOS_AT_UNREGISTER_ALL", "EXIT")
        if len(self.policy_replicas) == 0 and len(self.rollout_replicas) == 0:
            if (
                self.shut_down_event.is_set()
                or self.config.train.train_policy.type == "sft"
                or mode == "EXIT"
            ):
                logger.info("[Controller] Shutting down...")
                os.kill(os.getpid(), signal.SIGINT)
            if hasattr(self, "_first_policy_replica_arrived"):
                delattr(self, "_first_policy_replica_arrived")
            self.policy_init_done = False
            self.rollout_init_done = False

    async def post_register(self, atom: Atom, role: Role):
        # Update the desired number of replicas for policy and rollout if needed
        if role == Role.POLICY and not self.policy_init_done:
            if (
                len(self.policy_replicas)
                > self.config.policy.parallelism.n_init_replicas
            ):
                self.config.policy.parallelism.n_init_replicas = len(
                    self.policy_replicas
                )
                logger.info(
                    f"[Controller] Update policy n_init_replicas to {self.config.policy.parallelism.n_init_replicas} replicas"
                )
        elif role == Role.ROLLOUT and not self.rollout_init_done:
            if (
                len(self.rollout_replicas)
                > self.config.rollout.parallelism.n_init_replicas
            ):
                self.config.rollout.parallelism.n_init_replicas = len(
                    self.rollout_replicas
                )
                logger.info(
                    f"[Controller] Update rollout n_init_replicas to {self.config.rollout.parallelism.n_init_replicas} replicas"
                )

        # Check if all atoms of the replica have arrived
        if atom.replica.all_atoms_arrived:
            if atom.replica.start_time == -1:
                atom.replica.start_time = int(time.time())
            logger.info(
                f"[Controller] All atoms of {role} Replica {atom.replica.name} has been set."
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
                    # Exit and wait for the weight to be loaded with weight ready sent.
                await self.update_policies_initialize(valid_replicas, atom.replica)

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
                await self.update_rollouts_initialize(valid_replicas, atom.replica)
            else:
                raise Exception(f"[Controller] Unknown role during register: {role}")

    async def set_replica_ncclerror(self, replica_name: str, error: str):
        if replica_name in self.policy_replicas:
            self.policy_status_manager.set_ncclerror(replica_name, int(time.time()))

            # we use a time window to check nccl report, the last report will invoke post_ncclerror
            self.post_ncclerror_policy_invoke_id += 1
            current_invoke_id = self.post_ncclerror_policy_invoke_id
            await asyncio.sleep(constant.COSMOS_NCCL_ERROR_CLEAN_REPLICA_DELAY)
            if current_invoke_id == self.post_ncclerror_policy_invoke_id:
                # only the latest invoke will trigger the nccl error check
                await self.post_ncclerror(
                    self.policy_status_manager.get_all_policy_report_ncclerror(),
                    Role.POLICY,
                )
                self.policy_status_manager.clear_ncclerror()
        elif replica_name in self.rollout_replicas:
            raise NotImplementedError(
                f"[Controller] Rollout replica {replica_name} set timeout ack not supported"
            )
        else:
            logger.error(
                f"[Controller] Replica {replica_name} not found in both policy and rollout."
            )

    async def post_ncclerror(
        self, replicas_report_ncclerror: Dict[str, int], role: Role
    ):
        """
        This function is used to clean the hang replicas and trigger the buildmesh command
        """
        all_replicas_ = (
            self.policy_replicas if role == Role.POLICY else self.rollout_replicas
        )
        live_replicas = {rn: all_replicas_[rn] for rn in replicas_report_ncclerror}
        hang_replicas = [
            replica_name
            for replica_name in all_replicas_
            if replica_name not in live_replicas
        ]

        logger.info(f"[Controller] will clean hang replicas: {hang_replicas}")

        if len(live_replicas) == 1:
            # if there is only one replica, it's critical status, we should warning user to scale up the replica
            logger.warning(
                "[Controller] Only one replica is live, it's critical status, user should scale up the replica ASAP!"
            )

        # step 1, manual unregister the hang replicas, we only trigger buildmesh command after update the status
        manager = None
        if role == Role.POLICY:
            manager = self.policy_status_manager
            for hang_replica in hang_replicas:
                self.policy_replicas.pop(hang_replica)
                self.policy_status_manager.set_status(
                    hang_replica, PolicyStatus.DELETED
                )
        elif role == Role.ROLLOUT:
            raise NotImplementedError(
                f"[Controller] Rollout replica {hang_replica} set timeout ack not supported"
            )
        else:
            raise Exception(f"[Controller] Unknown role during post_ncclerror: {role}")

        # step 2, send buildmesh command to the live replicas
        if live_replicas:
            # Trigger Mesh building
            ranks = command.BuildMeshCommand.trigger(
                list(live_replicas.values()), redis_handler=self.redis_controller
            )
            manager.set_ranks(ranks)

    def rollout_statistic(
        self, sorted_replicas: List[Replica], train_step: int, total_steps: int
    ):
        rewards = []
        completion_lengths = []
        for _ in range(
            self.config.train.train_batch_per_replica * len(sorted_replicas)
        ):
            rollout = self.per_step_rollout_buffer.get()
            rewards.append(rollout.reward)
            completion_lengths.append(len(self.tokenizer.encode(rollout.completion)))
        report_data = {
            "train/reward_mean": np.mean(rewards),
            "train/reward_std": np.std(rewards),
            "train/reward_max": np.max(rewards),
            "train/reward_min": np.min(rewards),
            "train/completion_length_mean": np.mean(completion_lengths),
            "train/completion_length_max": np.max(completion_lengths),
        }

        if "wandb" in self.config.logging.logger and is_wandb_available():
            log_wandb(self.wandb_run, report_data, step=train_step)

        if "console" in self.config.logging.logger:
            logger.info(
                f"Step: {train_step}/{total_steps}, Reward Mean: {report_data['train/reward_mean']:.4f}, Reward Std: {report_data['train/reward_std']:.4f}, Reward Max: {report_data['train/reward_max']:.4f}, Reward Min: {report_data['train/reward_min']:.4f}, Completion Length Mean: {report_data['train/completion_length_mean']:.2f}, Completion Length Max: {report_data['train/completion_length_max']:.2f}"
            )
