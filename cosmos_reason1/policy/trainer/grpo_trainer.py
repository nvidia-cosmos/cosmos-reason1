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

from cosmos_reason1.policy.trainer import Trainer
from cosmos_reason1.policy.config import Config as CosmosConfig
import cosmos_reason1._cpp as cosmos_c
from cosmos_reason1.utils.parallelism import (
    ParallelDims,
    create_context_parallel_ctx,
)
import torch
import torch.nn.functional as F
import os
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.wandb_logger import is_wandb_available, log_wandb
from cosmos_reason1.utils.util import compute_mfu
import cosmos_reason1.utils.distributed as dist_util
import json
import time
import torch.distributed as dist
import numpy as np
import requests
import threading
import asyncio
from queue import Queue, Empty
from cosmos_reason1.dispatcher.command import (
    Command,
    BuildMeshCommand,
    PolicyToPolicyBroadcastCommand,
    PolicyToRolloutUnicastCommand,
    WeightResumeCommand,
    PolicyToPolicyUnicastCommand,
    DummyCommand,
    DataFetchCommand,
    AllReduceCommand,
    StopCommand,
)
import atexit
from cosmos_reason1.utils.util import (
    list_to_b64,
    b64_to_list,
    selective_log_softmax,
    msgpack_c_long,
    msgunpack_c_long,
    fix_data_type_size,
)
from cosmos_reason1.utils.parallelism_map import (
    ParallelTopoMapperGroup,
    slice_tensor_with_strategies,
)
from functools import cached_property
from qwen_vl_utils import process_vision_info
from typing import List, Callable
import types
from functools import partial
import msgpack


def compute_loss(
    current_token_logps: torch.Tensor,  # per-token logprobs of shape `[n_token_interested]`
    old_per_token_logps: torch.Tensor,  # per-token logprobs of shape `[n_token_interested]`
    ref_per_token_logps: torch.Tensor,  # per-token logprobs of shape `[n_token_interested]`
    current_advantages: torch.Tensor,  # of shape `[n_token_interested]`
    config: CosmosConfig,
) -> torch.Tensor:
    assert (
        current_token_logps.shape == current_advantages.shape
    ), "current_token_logps and current_advantages should have the same shape"
    assert (
        old_per_token_logps.shape == current_token_logps.shape
    ), "old_per_token_logps and ref_per_token_logps should have the same shape"
    # Compute the KL divergence between the model and the reference model
    if config.train.train_policy.kl_beta != 0.0:
        assert (
            not ref_per_token_logps.requires_grad
        ), "ref_per_token_logps should not require gradient"
        """
            With reference model used for KL. The logic should be further reviewed to verify.
        """
        per_token_kl = (
            torch.exp(ref_per_token_logps - current_token_logps)
            - (ref_per_token_logps - current_token_logps)
            - 1
        )

    coef_1 = torch.exp(current_token_logps - old_per_token_logps)
    coef_2 = torch.clamp(
        coef_1,
        1 - config.train.train_policy.epsilon_low,
        1 + config.train.train_policy.epsilon_high,
    )
    per_token_loss1 = coef_1 * current_advantages
    per_token_loss2 = coef_2 * current_advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    if config.train.train_policy.kl_beta != 0.0:
        """
            With reference model used for KL. The logic should be further reviewed to verify.
        """
        kl_loss = config.train.train_policy.kl_beta * per_token_kl
        per_token_loss += kl_loss

        return per_token_loss.mean(), kl_loss.mean()

    return per_token_loss.mean(), 0.0


class GRPOTrainer(Trainer):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims):
        super().__init__(config, parallel_dims)
        self.grpo_config = self.config.train.train_policy
        # For model load
        self.model_ready = False

        # For mesh build
        self.mesh_ready = False
        self.inter_policy_nccl = -1
        self.rollouts_comm = {}

        # For command fetch
        self.fetch_command_buffer = Queue()
        self.command_buffer = Queue()

        # For rollouts fetch
        self.data_queue = Queue()

        # For sync weight
        self.sync_weight_stream = torch.cuda.Stream(device=self.device)

        # Parallel parameters
        self.dp_rank, self.dp_world_size = 0, 1
        if parallel_dims.dp_enabled:
            self.dp_rank = parallel_dims.mesh["dp"].get_local_rank()
            self.dp_world_size = parallel_dims.mesh["dp"].size()

        # Init redis controller
        self.init_redis()

        # For iteration control
        self.total_num_steps = -1
        self.mini_step = 0
        self.train_step = 0
        self.optimize_step = 0
        self.batch_for_this_step = 0
        self.mini_batch = self.grpo_config.mini_batch

        # For Polocy to Rollout weight mapping
        self.parallel_mapper = None
        self.policy_to_rollout_insts = None

        # For GRPO
        self.max_length = config.policy.model_max_length
        self.is_cosmos = False
        if "cosmos" in self.grpo_config.dataset_name.lower():
            self.is_cosmos = True
            self.cosmos_cache_dir = os.environ.get(
                "COSMOS_CACHE", os.path.join(os.path.expanduser("~"), ".cache/cosmos/")
            )
            if not self.grpo_config.enable_dataset_preprocess:
                video_clips_path = os.path.join(
                    self.cosmos_cache_dir,
                    "datasets",
                    self.grpo_config.dataset_name,
                    self.grpo_config.dataset_subset,
                    "video_clips",
                )
                if not os.path.exists(video_clips_path):
                    raise FileNotFoundError(
                        f"Dataset directory {video_clips_path} does not exist. Please check the dataset path."
                    )
                mm_files_paths = {}
                for root, dirs, files in os.walk(video_clips_path):
                    for file in files:
                        if file.endswith(
                            (".mp4", ".avi", ".mov")
                        ):  # Common video extensions
                            mm_files_paths[file] = os.path.join(root, file)
                self.mm_files_paths = mm_files_paths
        if hasattr(self.hf_config, "image_token_id"):
            self.image_token = self.tokenizer.decode([self.hf_config.image_token_id])
            self.image_token_id = self.hf_config.image_token_id
        else:
            self.image_token = None
            self.image_token_id = None
        if hasattr(self.hf_config, "video_token_id"):
            self.video_token = self.tokenizer.decode([self.hf_config.video_token_id])
            self.video_token_id = self.hf_config.video_token_id
        else:
            self.video_token = None
            self.video_token_id = None
        # generate_every_optimize : Number of iterations per batch (denoted as μ in the algorithm).
        # Currently ony 1 is supported.
        self.mu_iterations = self.config.train.train_policy.mu_iterations
        self.optimizers.zero_grad()
        self.shutdown_background_task_event = threading.Event()
        self.fetch_command_thread = None
        self.fetch_rollouts_thread = None
        self.heartbeat_thread = self.start_heartbeat(
            self.shutdown_background_task_event
        )
        atexit.register(self.handle_shutdown)

    def handle_shutdown(self):
        self.shutdown_background_task_event.set()
        if self.global_rank == 0:
            if self.fetch_command_thread is not None:
                self.fetch_command_thread.join()
            if self.fetch_rollouts_thread is not None:
                self.fetch_rollouts_thread.join()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join()

    def model_load_from_hf(self):
        self.model.load_hf_weights(
            self.config.policy.model_name_or_path,
            self.parallel_dims,
            self.device,
        )
        self.model.train()
        self.model_ready = True

    def model_resume_from_checkpoint(self):
        self.ckpt_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizers,
            scheduler=self.lr_schedulers,
        )
        self.model.train()
        self.model_ready = True

    async def fetch_rollouts(self):
        assert self.global_rank == 0, "Only rank 0 can fetch rollouts"
        while not self.shutdown_background_task_event.is_set():
            rollouts = []
            try:
                rollouts = self.redis_controller.subscribe_rollout(self.replica_name)
            except Exception as e:
                logger.error(
                    f"Failed to get rollouts : {e} at replica {self.replica_name}"
                )
                raise e
            encountered_stop = False
            for r in rollouts:
                rollout = msgpack.unpackb(r)
                if (
                    "extra_info" in rollout
                    and rollout["extra_info"] is not None
                    and "is_end" in rollout["extra_info"]
                ):
                    assert rollout["extra_info"]["is_end"] in [True, "True", "true"]
                    encountered_stop = True
                self.data_queue.put_nowait(rollout)
            if encountered_stop:
                logger.info(
                    "[Policy] Encountered stop command in rollouts. Stopping the fetch loop."
                )
                break

    def check_inter_policy_all_ready(self):
        # Always return true for now due to pure local commands.
        return True

    def check_intra_policy_all_ready(self, command):
        commands = [DummyCommand for _ in range(self.world_size)]
        dist.all_gather_object(commands, command)
        is_mesh_builds = [isinstance(x, BuildMeshCommand) for x in commands]
        return all(is_mesh_builds)

    def execute_build_mesh(self, command: Command):
        if isinstance(command, BuildMeshCommand):
            if len(command.replica_name_to_rank) == 1:
                self.mesh_ready = True
                self.replica_name_to_rank = command.replica_name_to_rank
                return True
            else:
                if not self.check_inter_policy_all_ready():
                    return False
                assert self.replica_name in command.replica_name_to_rank
                rank = command.replica_name_to_rank[self.replica_name]
                mesh_key = self.get_group_unique_key(command.replica_name_to_rank)
                nccl_group_id = None
                max_retry = 3
                if rank == 0:
                    # initialize nccl handle for building mesh among policies
                    # only replica_rank == 0 have the right to generate nccl id.
                    nccl_group_id = cosmos_c.create_nccl_uid()
                    base64_nccl_group_id = list_to_b64(nccl_group_id)
                    while max_retry > 0:
                        try:
                            r = requests.post(
                                f"{self.remote_host}/api/nccl/comm_initiator",
                                json={
                                    "unique_pair_name": mesh_key,
                                    "handle_base64": base64_nccl_group_id,
                                },
                            )
                            if r.status_code != 200:
                                logger.error(
                                    f"[Policy] Error response in post nccl group_id to controller: {r.json()}"
                                )
                            else:
                                break
                        except Exception as e:
                            # just logging the error now.
                            logger.error(
                                f"[Policy] Failed in post nccl group_id to controller after {max_retry} retries: {str(e)}"
                            )
                        finally:
                            max_retry -= 1
                            if max_retry == 0:
                                raise RuntimeError(
                                    f"[Policy] Failed in post nccl group_id to controller after {max_retry} retries."
                                )
                            time.sleep(0.5)
                else:
                    # other replicas should query the nccl group id from controller
                    # all ranks need to wait for the rollout replica 0 finished the group_id post
                    # and then they can get the group_id from controller
                    # But we don't have something like dist.barrier(), so just while True loop to query it like synchronize.
                    max_retry = 1000
                    # all ranks not zero in replica 0 or all ranks of other replicas need to query the group_id from controller
                    while max_retry > 0:
                        try:
                            r = requests.post(
                                f"{self.remote_host}/api/nccl/comm_acceptor",
                                json={"unique_pair_name": mesh_key},
                            )
                            if r.status_code != 200:
                                pass
                            else:
                                base64_nccl_group_id = r.json()["handle_base64"]
                                nccl_group_id = b64_to_list(base64_nccl_group_id)
                                break
                        except Exception as e:
                            logger.error(
                                f"[Policy] Failed in query nccl group_id from controller: {str(e)}"
                            )
                        finally:
                            max_retry -= 1
                            if max_retry == 0:
                                raise RuntimeError(
                                    f"[Policy] Failed in query nccl group_id from controller after {max_retry} retries."
                                )
                            time.sleep(0.5)
                self.inter_policy_nccl = cosmos_c.create_nccl_comm(
                    nccl_group_id, rank, len(command.replica_name_to_rank)
                )
                logger.info(
                    f"[Policy] Created inter policy nccl comm {self.inter_policy_nccl} for rank {rank} with total {len(command.replica_name_to_rank)} ranks."
                )
                # Also need delete old nccl handler
                self.mesh_ready = True
                self.replica_name_to_rank = command.replica_name_to_rank
                return True
        return False

    def wrap_to_cuda_tensor(self, key, obj, in_place=False):
        """
        wrap the object to cuda tensor for sync parameters using nccl.
        """
        if isinstance(obj, torch.distributed.tensor.DTensor):
            assert (
                obj.device == self.device
            ), "DTensor is not on the same device as the model."
            return obj.to_local()
        elif isinstance(obj, torch.Tensor):
            if obj.device != self.device:
                if in_place:
                    raise ValueError(
                        f"Object {key} is not on the same device as the model. Please set in_place to False."
                    )
                obj = obj.to(self.device)
            return obj
        elif isinstance(obj, np.ndarray):
            if in_place:
                raise ValueError(
                    f"Object {key} is not a tensor. Please set in_place to False."
                )
            obj = torch.from_numpy(obj).to(self.device)
            return obj
        else:
            if in_place:
                raise ValueError(
                    f"Object {key} is not a tensor. Please set in_place to False."
                )
            if isinstance(obj, tuple):
                obj = tuple(
                    [x.tolist() if isinstance(x, np.ndarray) else x for x in obj]
                )
                obj = fix_data_type_size(obj)
            bytes = msgpack.packb(obj, default=msgpack_c_long)
            obj = torch.frombuffer(bytes, dtype=torch.uint8).to(self.device)
            return obj

    def extract_from_cuda_tensor(self, key, obj, tensor):
        """
        Extract the object from cuda tensor for sync parameters using nccl.
        """
        if isinstance(obj, torch.distributed.tensor.DTensor):
            assert (
                obj.device == self.device
            ), "DTensor is not on the same device as the model."
        elif isinstance(obj, torch.Tensor):
            if obj.device != self.device:
                obj.copy_(tensor)
        elif isinstance(obj, np.ndarray):
            if obj.shape != tensor.shape:
                raise ValueError(
                    f"Object {key} is not the same shape as the tensor. Please check the data consistency."
                )
            x = tensor.cpu()
            obj.copy_(x.numpy())
        else:
            np_arr = tensor.cpu()
            obj_new = msgpack.unpackb(bytes(np_arr.numpy()), ext_hook=msgunpack_c_long)
            if isinstance(obj, tuple):
                assert len(obj) == len(obj_new)
                obj = tuple(
                    [
                        np.array(obj_new[idx])
                        if isinstance(x, np.ndarray)
                        else tuple(obj_new[idx])
                        if isinstance(x, tuple)
                        else obj_new[idx]
                        for idx, x in enumerate(obj)
                    ]
                )
        return obj

    def sync_all_states(self, is_send: bool, send_hook: callable, recv_hook: callable):
        """
        Sync all states of the model and optimizer.
        """
        len_params = 0
        self_state_dict = self.model.state_dict()
        for dest_name in sorted(self_state_dict.keys()):
            obj = self_state_dict[dest_name]
            local_view = self.wrap_to_cuda_tensor(dest_name, obj, in_place=True)
            if is_send:
                # nccl send
                send_hook(local_view)
            else:
                # nccl recv
                recv_hook(local_view)
            len_params += 1
        optimizer_state = self.optimizers.state_dict()
        for dest_name in sorted(optimizer_state.keys()):
            obj = optimizer_state[dest_name]
            local_view = self.wrap_to_cuda_tensor(dest_name, obj)
            if is_send:
                # nccl send
                send_hook(local_view)
            else:
                # nccl recv
                recv_hook(local_view)
                optimizer_state[dest_name] = self.extract_from_cuda_tensor(
                    dest_name, obj, local_view
                )
            len_params += 1
        if not is_send:
            self.optimizers.load_state_dict(optimizer_state)
        lr_sheduler_state = self.lr_schedulers.state_dict()
        for dest_name in sorted(lr_sheduler_state.keys()):
            obj = lr_sheduler_state[dest_name]
            local_view = self.wrap_to_cuda_tensor(dest_name, obj)
            if is_send:
                # nccl send
                send_hook(local_view)
            else:
                # nccl recv
                recv_hook(local_view)
                lr_sheduler_state[dest_name] = self.extract_from_cuda_tensor(
                    dest_name, obj, local_view
                )
            len_params += 1
        if not is_send:
            self.lr_schedulers.load_state_dict(lr_sheduler_state)
        rng_state = self.ckpt_manager.get_rng_state()
        for dest_name in sorted(rng_state.keys()):
            obj = rng_state[dest_name]
            local_view = self.wrap_to_cuda_tensor(dest_name, obj)
            if is_send:
                # nccl send
                send_hook(local_view)
            else:
                # nccl recv
                recv_hook(local_view)
                rng_state[dest_name] = self.extract_from_cuda_tensor(
                    dest_name, obj, local_view
                )
            len_params += 1
        if not is_send:
            self.ckpt_manager.set_rng_state(rng_state)
        return len_params

    def execute_policy_to_policy_broadcast(
        self, command: PolicyToPolicyBroadcastCommand
    ):
        assert self.mesh_ready, "Mesh must be built before policy to policy broadcast"
        send = self.replica_name == command.src_replica_name
        recv = self.replica_name in command.dst_replica_names and not send
        assert command.src_replica_name in self.replica_name_to_rank
        src_rank = self.replica_name_to_rank[command.src_replica_name]
        if not send and not recv:
            return True
        st = time.time()
        send_recv_hook = partial(
            cosmos_c.nccl_broadcast, rank=src_rank, comm_idx=self.inter_policy_nccl
        )
        len_params = self.sync_all_states(
            is_send=send,
            send_hook=send_recv_hook,
            recv_hook=send_recv_hook,
        )
        if recv:
            self.model_ready = True
        time_eclapsed = time.time() - st
        logger.debug(
            f"[Policy] Policy2Policy Broadcast {len_params} parameters from {command.src_replica_name} (rank {src_rank}) to {len(command.dst_replica_names)} replicas took {time_eclapsed:.3f} seconds."
        )
        return True

    def execute_policy_to_policy_unicast(self, command: PolicyToPolicyUnicastCommand):
        assert self.mesh_ready
        send = self.replica_name == command.src_replica_name
        recv = self.replica_name == command.dst_replica_name
        assert command.src_replica_name in self.replica_name_to_rank
        assert command.dst_replica_name in self.replica_name_to_rank
        src_rank = self.replica_name_to_rank[command.src_replica_name]
        dst_rank = self.replica_name_to_rank[command.dst_replica_name]
        if not send and not recv:
            return True
        st = time.time()
        send_hook = partial(
            cosmos_c.nccl_send, peer=dst_rank, comm_idx=self.inter_policy_nccl
        )
        recv_hook = partial(
            cosmos_c.nccl_recv, peer=src_rank, comm_idx=self.inter_policy_nccl
        )
        len_params = self.sync_all_states(
            is_send=send,
            send_hook=send_hook,
            recv_hook=recv_hook,
        )
        if recv:
            self.model_ready = True
        time_eclapsed = time.time() - st
        logger.debug(
            f"[Policy] Policy2Policy Unicast {len_params} parameters from {command.src_replica_name} (rank {src_rank}) to {command.dst_replica_name} (rank {dst_rank}) as sender {send} took {time_eclapsed:.3f} seconds."
        )
        return True

    @cached_property
    def map_w_from_policy_to_rollout(self):
        """
        Generate a mapping from local parameters into shape/layout that rollout requires.
        The mapping is created by iterating through the named parameters of both models
        and replacing certain substrings in the parameter names.
        """
        name_to_transform = {}
        assert len(self.model.sorted_params) > 0, "No sorted parameters found."
        for name, shape in self.model.sorted_params:
            transform_block = self.model.weight_sync_transform_by_key(name)
            # Condition is relaxed from shape matching to number of elements matching because the tensor may be transposed/reshaped
            if isinstance(transform_block, Callable):
                mapped_tensor = transform_block()
            elif isinstance(transform_block, torch.Tensor):
                mapped_tensor = transform_block
            else:
                raise ValueError(
                    f"Transform block is not a callable or tensor: {transform_block}"
                )
            assert (
                mapped_tensor.nelement() == int(np.prod(shape))
            ), f"Number of elements mismatch: {mapped_tensor.nelement()} != {np.prod(shape)} for {name}"
            name_to_transform[name] = transform_block
        return name_to_transform

    def execute_policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        assert command.src_replica_size == self.world_size
        if self.parallel_mapper is None:
            self.parallel_mapper = ParallelTopoMapperGroup(
                self.config.policy.parallelism,
                self.config.rollout.parallelism,
                self.world_size,
                command.dst_replica_size,
                self.hf_config,
                self.config.policy.model_name_or_path,
            )
        send = command.src_replica_name == self.replica_name
        if not send:
            return True
        comm_id = -1
        mesh_key = command.src_replica_name + "_" + command.dst_replica_name
        if mesh_key in self.rollouts_comm:
            comm_id = self.rollouts_comm[mesh_key]
        else:
            nccl_uuid = [None]
            if self.global_rank == 0:
                # initialize nccl handle for building mesh among policies
                # Only create nccl group id in rank 0.
                nccl_uuid[0] = cosmos_c.create_nccl_uid()
                max_retry = 3
                base64_nccl_group_id = list_to_b64(nccl_uuid[0])
                logger.debug(f"[Policy] mesh_key: {mesh_key}")
                while max_retry > 0:
                    try:
                        r = requests.post(
                            f"{self.remote_host}/api/nccl/comm_initiator",
                            json={
                                "unique_pair_name": mesh_key,
                                "handle_base64": base64_nccl_group_id,
                            },
                        )
                        if r.status_code != 200:
                            logger.error(
                                f"[Policy] Error response in post nccl group_id to controller: {r.json()}"
                            )
                        else:
                            break
                    except Exception as e:
                        # just logging the error now.
                        logger.error(
                            f"[Policy] Failed in post nccl group_id to controller after {max_retry} retries: {str(e)}"
                        )
                    finally:
                        max_retry -= 1
                        if max_retry == 0:
                            raise RuntimeError(
                                f"[Policy] Failed in post nccl group_id to controller after {max_retry} retries."
                            )
                        time.sleep(0.5)
            # broadcast the nccl group id to all ranks
            if self.world_size > 1:
                dist.broadcast_object_list(nccl_uuid, src=0, device=torch.device("cpu"))
            comm_id = cosmos_c.create_nccl_comm(
                nccl_uuid[0],
                self.global_rank,
                self.world_size + command.dst_replica_size,
            )
            logger.info(
                f"[Policy] Create policy to rollout nccl comm: {comm_id} for {mesh_key}"
            )
            self.rollouts_comm[mesh_key] = comm_id
            assert (
                self.map_w_from_policy_to_rollout is not None
            ), "No parameters to sync found."
        # Check the model parameters for sync consistency
        # This is a sanity check to make sure the model parameters are consistent
        # Commenting out for now, since it is time consuming and not necessary
        # for name, shape in self.model.sorted_params():
        #     local_view = self.model.weight_sync_transform_by_key(name)
        #     sync_view = self.get_parameter_to_sync(name)
        #     assert local_view.data_ptr() == sync_view.data_ptr(), (
        #         f"Data pointer mismatch: {local_view.data_ptr()} != {sync_view.data_ptr()} for {name}"
        #     )
        #     assert local_view.shape == shape, (
        #         f"Shape mismatch: {local_view.shape} != {shape} for {name}"
        #     )
        #     assert sync_view.shape == shape, (
        #         f"Shape mismatch: {sync_view.shape} != {shape} for {name}"
        #     )
        #     assert local_view.dtype == sync_view.dtype, (
        #         f"Data type mismatch: {local_view.dtype} != {sync_view.dtype} for {name}"
        #     )
        st = time.time()
        # sort the param list by the dest_name, same as rollout
        if self.policy_to_rollout_insts is None:
            param = self.model.sorted_params
            insts = self.parallel_mapper.generate_policy_to_rollout_insts(
                param, self.global_rank
            )
            self.policy_to_rollout_insts = insts
        total_bytes_sent = 0
        with torch.cuda.stream(self.sync_weight_stream):
            for inst in self.policy_to_rollout_insts:
                p_rank, r_rank, tensor_split_strategys, dest_name, shape = inst
                if dest_name not in self.map_w_from_policy_to_rollout:
                    raise Exception(
                        f"[Policy] {dest_name} not in map_w_from_policy_to_rollout. Please call execute_policy_to_rollout_unicast_preparation first."
                    )
                local_view = self.map_w_from_policy_to_rollout[dest_name]
                if isinstance(local_view, Callable):
                    local_view = local_view()
                assert (
                    local_view.nelement() == int(np.prod(shape))
                ), f"Number of elements mismatch: {local_view.nelement()} != {np.prod(shape)} for {dest_name}"
                view = slice_tensor_with_strategies(
                    local_view, tensor_split_strategys
                ).contiguous()
                assert self.global_rank == p_rank
                # send local view to rollout r_rank
                # logger.info(
                #     f"[Policy] rank {self.global_rank} send tensor: {dest_name} to rank {r_rank + command.src_replica_size} with shape: {view.shape} out of {local_view.shape}"
                # )
                cosmos_c.nccl_send(view, r_rank + command.src_replica_size, comm_id)
                total_bytes_sent += view.numel() * view.element_size()
        # make sure all the send operations of all ranks are finished
        time_eclapsed = time.time() - st
        logger.debug(
            f"[Policy] All {len(self.policy_to_rollout_insts)} send operations of finished in {time_eclapsed:.3f} seconds with {total_bytes_sent / (1024 * 1024)} MB sent."
        )
        return True

    def execute_weight_resume(self, command: WeightResumeCommand = None):
        if self.config.train.resume:
            try:
                self.model_resume_from_checkpoint()
            except Exception as e:
                if isinstance(e, FileNotFoundError):
                    logger.info(
                        f"Fail to resume from {self.config.train.resume} because the checkpoint file does not exist, trying to load from HuggingFace..."
                    )
                else:
                    logger.error(
                        f"Cannot resume from {self.config.train.resume} {e}. Trying to load from HuggingFace..."
                    )
                self.model_load_from_hf()
        else:
            logger.info("Resume not set. Trying to load from HuggingFace...")
            self.model_load_from_hf()
        logger.info("[Policy] Model loaded from checkpoint.")
        return True

    def train_ack(self):
        if self.global_rank == 0:
            max_retry = 3
            while max_retry > 0:
                try:
                    r = requests.post(
                        f"{self.remote_host}/api/policy/train_ack",
                        json={
                            "replica_name": self.replica_name,
                            "iteration_count": self.train_step,
                        },
                    )
                    if r.status_code != 200:
                        logger.error(
                            f"[Policy] Error response in send train ack: {r.json()}"
                        )
                    else:
                        break
                except Exception as e:
                    logger.error(
                        f"[Policy] Failed to send train ack : {e} at replica {self.replica_name}"
                    )
                finally:
                    max_retry -= 1
                    if max_retry == 0:
                        raise RuntimeError(
                            f"[Policy] Failed in in send train ack to controller after {max_retry} retries."
                        )
                    time.sleep(0.5)

    def execute_data_fetch(self, command: DataFetchCommand):
        assert self.replica_name == command.replica_name
        self.batch_for_this_step = command.items_count
        self.train()
        self.train_ack()
        logger.debug(f"[Policy] Train ack sent for global step {command.global_step}.")
        return True

    def execute_all_reduce(self, command: AllReduceCommand = None):
        """
        # Add nccl allreduce operations for all parameters and necessary states.
        """
        for model_part in self.model_parts:
            # Do allreduce of gradient in all policy replicas.
            dist_util.gradient_reduce_across_dp_replicas_(
                [p for p in model_part.parameters()], self.inter_policy_nccl
            )

            # Then clipping gradient norm
            dist_util.gradient_norm_clipping(
                [p for p in model_part.parameters()],
                self.config.train.optm_grad_norm_clip,
                foreach=True,
                pp_mesh=self.parallel_dims.mesh["pp"]
                if self.parallel_dims.pp_enabled
                else None,
            )
        self.train_stream.wait_stream(self.sync_weight_stream)
        self.optimizers.step()
        self.lr_schedulers.step()
        logger.debug(
            f"[Policy] Optimization step {self.optimize_step + 1} at train step {self.train_step + 1} finished."
        )
        self.optimize_step += 1
        self.optimizers.zero_grad()
        return True

    async def fetch_command(self):
        assert self.global_rank == 0, "Only rank 0 can fetch command"
        while not self.shutdown_background_task_event.is_set():
            commands = []
            try:
                commands = self.redis_controller.subscribe_command(self.replica_name)
            except Exception as e:
                logger.error(
                    f"Failed to get commands : {e} at replica {self.replica_name}"
                )
                raise e
            try:
                encountered_stop = False
                for x in commands:
                    command = Command.depack(x)
                    if isinstance(command, StopCommand):
                        encountered_stop = True
                    self.fetch_command_buffer.put_nowait(command)
                if encountered_stop:
                    logger.info("[Policy] Stop command received. Exiting...")
                    break
            except Exception as e:
                logger.error(e)
                raise e

    def execute_command(self, command: Command):
        logger.debug(f"[Policy] Process command {command._serialize()}")
        command_done = self.execute_build_mesh(command)
        if not isinstance(command, BuildMeshCommand):
            if isinstance(command, WeightResumeCommand):
                command_done = self.execute_weight_resume(command)
            elif isinstance(command, PolicyToRolloutUnicastCommand):
                command_done = self.execute_policy_to_rollout_unicast(command)
            elif isinstance(command, PolicyToPolicyBroadcastCommand):
                command_done = self.execute_policy_to_policy_broadcast(command)
            elif isinstance(command, PolicyToPolicyUnicastCommand):
                command_done = self.execute_policy_to_policy_unicast(command)
            elif isinstance(command, DataFetchCommand):
                command_done = self.execute_data_fetch(command)
            elif isinstance(command, StopCommand):
                logger.info("[Policy] Stop command received. Exiting...")
                self.handle_shutdown()
                command_done = True
            else:
                raise Exception(f"No such command supoorted in policy {command}")
        logger.debug(
            f"[Policy] Command {command._serialize()} executed: {command_done}"
        )
        return command_done

    def broadcast_command(self):
        command = [list()]
        if self.global_rank == 0:
            while len(self.fetch_command_buffer.queue) > 0:
                command[0].append(self.fetch_command_buffer.get_nowait())
        dist.broadcast_object_list(command, src=0, device=torch.device("cpu"))
        if len(command[0]) > 0:
            for c in command[0]:
                self.command_buffer.put_nowait(c)

    def main_loop(self):
        def fetch_command_helper(trainer: GRPOTrainer):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(trainer.fetch_command())
            new_loop.stop()
            new_loop.close()
            return

        def fetch_rollouts_helper(trainer: GRPOTrainer):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(trainer.fetch_rollouts())
            new_loop.stop()
            new_loop.close()
            return

        if self.global_rank == 0:
            # Start the thread with daemon=True, so it will exit when the main program exits.
            self.fetch_command_thread = threading.Thread(
                target=fetch_command_helper, args=(self,), daemon=True
            ).start()
            self.fetch_rollouts_thread = threading.Thread(
                target=fetch_rollouts_helper, args=(self,), daemon=True
            ).start()

        while not self.shutdown_background_task_event.is_set():
            self.broadcast_command()
            while len(self.command_buffer.queue) > 0:
                cmd = self.command_buffer.get_nowait()
                assert self.execute_command(cmd)
        logger.info("[Policy] Main loop finished. Shutdown background task event set.")

    def dispatch_rollouts(self):
        rollouts = [[]]
        scattered_rollouts = [[] for _ in range(self.world_size)]
        self.batch_for_this_step = (
            self.batch_for_this_step // self.dp_world_size * self.dp_world_size
        )
        assert self.batch_for_this_step % self.dp_world_size == 0
        if self.global_rank == 0:
            dp_id = 0
            for _ in range(self.batch_for_this_step):
                try:
                    rollout = self.data_queue.get(block=True, timeout=None)
                except Empty:
                    logger.error(
                        "[Policy] Rollouts queue is empty, please check the dispatcher."
                    )
                    raise Empty
                for i in range(self.world_size):
                    if self.parallel_dims.get_rank_in_dim("dp", i) == dp_id:
                        scattered_rollouts[i].append(rollout)
                        # logger.info(f"[Policy] Rollout {dp_id} dispatched to rank {i}, dp world_size {self.dp_world_size}")
                dp_id += 1
                if dp_id >= self.dp_world_size:
                    dp_id = 0
        if self.world_size == 1:
            return scattered_rollouts[0]
        dist.scatter_object_list(
            rollouts,
            scattered_rollouts,
            src=0,
        )
        return rollouts[0]

    def _get_per_token_logps(self, input, full_logits) -> torch.Tensor:
        input_ids_batch = input["input_ids"]
        logprob_masks = input.pop("logprob_masks")
        token_masks = input.pop("token_masks")
        advantages = input.pop("advantages")[token_masks]
        logits = full_logits[logprob_masks]
        input_ids_of_interest = input_ids_batch[token_masks]
        assert (
            logits.shape[0] == input_ids_of_interest.shape[0]
        ), f"Logits shape {logits.shape} does not match input_ids shape {input_ids_of_interest.shape}"
        if self.config.train.train_policy.temperature > 1e-6:
            logits = logits / self.config.train.train_policy.temperature
        logps = selective_log_softmax(logits, input_ids_of_interest)
        return logps, advantages

    def gen_cosmos_sample(self, sample):
        assert "video" in sample or "image" in sample, "No video/image in the sample"

        choices = sample["qa_pairs"]["index2ans"]
        system_prompt = self.grpo_config.system_prompt

        user_prompt = (
            sample["qa_pairs"]["question"]
            + "\n"
            + "\n".join([f"({i}) {choice}" for i, choice in choices.items()])
        )
        user_prompt += (
            "\nAnswer with the option's letter from the given choices directly."
        )
        user_prompt += "\nPlease answer the question in the following format: <think> your reasoning </think> <answer> your answer </answer>."

        if self.grpo_config.enable_dataset_preprocess:
            if "video" in sample:
                asset = sample["video"]
                placeholder = "video"
            else:
                asset = sample["image"]
                placeholder = "image"
            multi_modal_content = {
                "type": placeholder,
                placeholder: "",
            }

            # TODO(dinghaoy): Currently does not support multiple videos or images,
            # since the cosmos dataset only has one video each sample
            video_tensors_path = os.path.join(
                self.cosmos_cache_dir,
                "datasets",
                self.grpo_config.dataset_name,
                self.grpo_config.dataset_subset,
                "video_tensors",
                f"fps-{self.grpo_config.fps}-pixels-{self.grpo_config.max_pixels}",
            )
            asset = os.path.basename(asset).split(".")[0]
            asset_path = os.path.join(video_tensors_path, f"{asset}.cosmos")
            assert os.path.exists(asset_path), f"Asset {asset_path} does not exist"
            loaded_asset = torch.load(asset_path, map_location="cpu")
            n_tokens = loaded_asset[f"n_{placeholder}_tokens"]

            conversations = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        multi_modal_content,
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ]

            text = self.hf_processor.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=False
            )
            temp_image_token = "[COSMOS_IMAGE_TOKEN]"
            temp_video_token = "[COSMOS_VIDEO_TOKEN]"

            if placeholder == "image":
                text = text.replace(self.image_token, n_tokens * temp_image_token, 1)
            elif placeholder == "video":
                text = text.replace(self.video_token, n_tokens * temp_video_token, 1)

            text = text.replace(temp_image_token, self.image_token)
            text = text.replace(temp_video_token, self.video_token)

            encoded_text = self.tokenizer.encode(text, add_special_tokens=False)

            result_dict = {
                "input_ids": encoded_text[: self.max_length],
            }

            if placeholder == "video":
                result_dict["pixel_values_videos"] = loaded_asset["pixel_values_videos"]
                result_dict["video_grid_thw"] = loaded_asset["video_grid_thw"]
                result_dict["second_per_grid_ts"] = torch.tensor(
                    loaded_asset["second_per_grid_ts"], dtype=torch.float
                )
                result_dict["pixel_values_videos_lengths_per_sample"] = result_dict[
                    "pixel_values_videos"
                ].shape[0]
            else:  # image
                result_dict["pixel_values_images"] = loaded_asset["pixel_values_images"]
                result_dict["image_grid_thw"] = loaded_asset["image_grid_thw"]
                result_dict["pixel_values_images_lengths_per_sample"] = result_dict[
                    "pixel_values_images"
                ].shape[0]
        else:
            if "video" in sample:
                multi_modal_content = {
                    "type": "video",
                    "video": self.mm_files_paths[sample["video"].split("/")[-1]],
                    "max_pixels": self.grpo_config.max_pixels,
                    "fps": self.grpo_config.fps,
                }
            else:
                multi_modal_content = {
                    "type": "image",
                    "image": self.mm_files_paths[sample["image"].split("/")[-1]],
                }
            conversations = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        multi_modal_content,
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ]
            prompt = self.hf_processor.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
            )
            image_inputs, video_inputs = process_vision_info(conversations)
            inputs = self.hf_processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            result_dict = {
                "input_ids": inputs["input_ids"][0].tolist()[: self.max_length],
                "pixel_values_videos": inputs["pixel_values_videos"],
                "video_grid_thw": inputs["video_grid_thw"],
                "second_per_grid_ts": torch.tensor(
                    inputs["second_per_grid_ts"], dtype=torch.float
                ),
                "pixel_values_videos_lengths_per_sample": inputs[
                    "pixel_values_videos"
                ].shape[0],
            }

        return result_dict

    def batch_cosmos_sample(self, samples):
        pixel_values_videos = []
        video_grid_thw = []
        second_per_grid_ts = []
        pixel_values_images = []
        image_grid_thw = []
        pixel_values_videos_lengths_per_sample = []
        pixel_values_images_lengths_per_sample = []

        for x in samples:
            if "pixel_values_videos" in x:
                pixel_values_videos.append(x["pixel_values_videos"])
                video_grid_thw.append(x["video_grid_thw"])
                second_per_grid_ts.append(x["second_per_grid_ts"])
                pixel_values_videos_lengths_per_sample.append(
                    x["pixel_values_videos_lengths_per_sample"]
                )
            if "pixel_values_images" in x:
                pixel_values_images.append(x["pixel_values_images"])
                image_grid_thw.append(x["image_grid_thw"])
                pixel_values_images_lengths_per_sample.append(
                    x["pixel_values_images_lengths_per_sample"]
                )

        if len(pixel_values_videos) > 0:
            max_len = max([x.shape[0] for x in pixel_values_videos])
            for i in range(len(pixel_values_videos)):
                pixel_values_videos[i] = pixel_values_videos[i].unsqueeze(0)
                assert (
                    pixel_values_videos[i].ndim == 3
                ), f"pixel_values_videos[i].ndim: {pixel_values_videos[i].ndim}"
                pixel_values_videos[i] = F.pad(
                    pixel_values_videos[i],
                    (0, 0, 0, max_len - pixel_values_videos[i].shape[1]),
                )
            pixel_values_videos = torch.cat(pixel_values_videos, dim=0)
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
            second_per_grid_ts = torch.cat(second_per_grid_ts, dim=0)
        if len(pixel_values_images) > 0:
            max_len = max([x.shape[0] for x in pixel_values_images])
            for i in range(len(pixel_values_images)):
                pixel_values_images[i] = pixel_values_images[i].unsqueeze(0)
                assert (
                    pixel_values_images[i].ndim == 3
                ), f"pixel_values_images[i].ndim: {pixel_values_images[i].ndim}"
                pixel_values_images[i] = F.pad(
                    pixel_values_images[i],
                    (0, 0, 0, max_len - pixel_values_images[i].shape[1]),
                )
            image_grid_thw = torch.cat(image_grid_thw, dim=0)

        batch = {}

        if len(pixel_values_videos) > 0:
            batch["pixel_values_videos"] = pixel_values_videos
            batch["video_grid_thw"] = video_grid_thw
            batch["second_per_grid_ts"] = second_per_grid_ts
            batch["pixel_values_videos_lengths_per_sample"] = torch.tensor(
                pixel_values_videos_lengths_per_sample, dtype=torch.long
            ).view(-1, 1)
        if len(pixel_values_images) > 0:
            batch["pixel_values_images"] = pixel_values_images
            batch["image_grid_thw"] = image_grid_thw
            batch["pixel_values_images_lengths_per_sample"] = torch.tensor(
                pixel_values_images_lengths_per_sample, dtype=torch.long
            ).view(-1, 1)

        return batch

    def batch_encoding(
        self, input_texts: List[str], completion_texts: List[str], is_cosmos=False
    ):
        if is_cosmos:
            samples = [
                self.gen_cosmos_sample(json.loads(input_text))
                for input_text in input_texts
            ]
            encoded_inputs = self.batch_cosmos_sample(samples)
            input_ids = [x["input_ids"] for x in samples]
        else:
            input_ids = self.tokenizer(
                input_texts, truncation=True, padding=False
            ).input_ids  # Don't pad yet
            encoded_inputs = {}

        completion_ids = self.tokenizer(
            completion_texts, truncation=True, padding=False
        ).input_ids  # Don't pad yet

        merged_ids = []
        logprob_mask = []
        token_mask = []
        for x, y in zip(input_ids, completion_ids):
            merged_ids.append(x + y)
            logprob_mask.append([0] * (len(x) - 1) + [1] * len(y) + [0])
            token_mask.append([0] * len(x) + [1] * len(y))
        return encoded_inputs, merged_ids, logprob_mask, token_mask

    def train(self):
        pp_last_stage = (
            self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
        )

        # Do it once
        if (
            pp_last_stage
            and self.parallel_dims.pp_enabled
            and not hasattr(self, "swizzled_forward")
        ):
            # Swizzle the forward function to return the current per-token logprobs.
            orig_forward = self.model.forward
            self.model.forward = types.MethodType(
                partial(
                    _swizzle_pp_grpo_forward,
                    self,
                    orig_forward,
                    self.config,
                ),
                self.model,
            )
            self.swizzled_forward = True

        start_time = time.time()
        logger.debug("[Policy] Prepare training data.")
        rollouts = self.dispatch_rollouts()
        ### Fake
        prompts = [rollout["prompt"] for rollout in rollouts]
        labels = [rollout["completion"] for rollout in rollouts]
        # logger.info(f"prompts shape: {len(prompts)}, labels shape: {len(labels)}")
        # for i in range(len(prompts)):
        #     logger.info(f"prompts[{i}] shape: {len(prompts[i])}, labels[{i}] shape: {len(labels[i])}")
        advantage = [rollout["advantage"] for rollout in rollouts]
        # print(f"advantage shape: {len(advantage), advantage}")
        # prompt_idx = [rollout["prompt_idx"] for rollout in rollouts]
        # extra_info = [rollout["extra_info"] for rollout in rollouts]
        kwargs, merged_ids, logprob_masks, token_masks = self.batch_encoding(
            prompts, labels, is_cosmos=self.is_cosmos
        )
        kwargs = {name: tensor.to(self.device) for name, tensor in kwargs.items()}

        inputs = {
            "input_ids": merged_ids,
            "logprob_masks": logprob_masks,
            "token_masks": token_masks,
            "advantages": torch.tensor(advantage).to(self.device),
            # TODO(lfeng): To be added for reference model
            # "ref_per_token_logps": torch.tensor(ref_per_token_logps).to(self.device),
        }
        inputs.update(kwargs)
        logger.debug(
            f"[Policy] Start training with prompts {len(inputs['input_ids'])}."
        )
        # Currently, we only support no cp parallelism for policy training.
        assert not self.parallel_dims.cp_enabled
        batch_size = len(inputs["input_ids"])
        mini_batch_size = (
            min(self.mini_batch, batch_size) if self.mini_batch > 0 else batch_size
        )
        assert (
            batch_size % mini_batch_size == 0
        ), "Batch size should be divided evenly by mini_batch"
        num_mini_batch = batch_size // mini_batch_size
        self.old_per_token_logps = [None for _ in range(num_mini_batch)]

        # Validate the PP parallelism configuration
        if self.parallel_dims.pp_enabled:
            n_microbatches = (
                mini_batch_size // self.config.policy.parallelism.pp_micro_batch_size
            )
            assert (
                n_microbatches % self.parallel_dims.pp == 0
            ), f"n_microbatches {n_microbatches} should be divided evenly by pp size of {self.parallel_dims.pp}"

        for i_mu in range(self.mu_iterations):
            local_mini_step = 0
            with torch.cuda.stream(self.train_stream):
                for i in range(0, batch_size, mini_batch_size):
                    end = min(i + mini_batch_size, batch_size)
                    input = {}
                    for key in inputs:
                        input[key] = inputs[key][i:end]

                    # TODO(jiaxin): support variable length in PP
                    max_len = (
                        self.config.policy.model_max_length
                        if self.parallel_dims.pp_enabled
                        else max([len(x) for x in input["input_ids"]])
                    )
                    max_len = (
                        (max_len + self.seq_len_multiple - 1)
                        // self.seq_len_multiple
                        * self.seq_len_multiple
                    )
                    input["input_ids"] = torch.tensor(
                        [
                            x
                            + [self.tokenizer.pad_token_id] * (max(0, max_len - len(x)))
                            for x in input["input_ids"]
                        ],
                        dtype=torch.long,
                    ).to(self.device)
                    input["logprob_masks"] = torch.tensor(
                        [
                            x + [0] * (max(0, max_len - len(x)))
                            for x in input["logprob_masks"]
                        ],
                        dtype=torch.bool,
                    ).to(self.device)
                    input["token_masks"] = torch.tensor(
                        [
                            x + [0] * (max(0, max_len - len(x)))
                            for x in input["token_masks"]
                        ],
                        dtype=torch.bool,
                    ).to(self.device)

                    # [batch_size] -> [batch_size, max_len] via expanding
                    input["advantages"] = (
                        input["advantages"].unsqueeze(1).expand(-1, max_len)
                    )

                    position_ids, pos_seq_dim = self.model.get_position_ids(**input)
                    input["position_ids"] = position_ids
                    cp_context = (
                        create_context_parallel_ctx(
                            cp_mesh=self.parallel_dims.mesh["cp"],
                            cp_buffers=[input["input_ids"], input["position_ids"]],
                            cp_seq_dims=[1, pos_seq_dim],
                            cp_no_restore_buffers={
                                input["input_ids"],
                                input["position_ids"],
                            },
                            cp_rotate_method=self.config.policy.parallelism.cp_rotate_method,
                        )
                        if self.parallel_dims.cp_enabled
                        else None
                    )
                    with self.context(cp_context):
                        if self.parallel_dims.pp_enabled:
                            if pp_last_stage:
                                if self.old_per_token_logps[local_mini_step] is None:
                                    assert (
                                        i_mu == 0
                                    ), "Only first `mu_iteration` should append `old_per_token_logps`"
                                else:
                                    assert (
                                        i_mu > 0
                                    ), "Only `mu_iteration > 0` should reuse `old_per_token_logps`"
                                    assert (
                                        len(self.old_per_token_logps[local_mini_step])
                                        == n_microbatches
                                    )

                            # [mini_batch_size, 1]: indicating the index of mini-batch
                            mini_batch_ids_cpu = torch.Tensor(
                                [[local_mini_step]] * mini_batch_size
                            ).int()
                            micro_batch_ids_list = []
                            for i in range(mini_batch_size):
                                micro_batch_ids_list.append(
                                    [
                                        i
                                        // self.config.policy.parallelism.pp_micro_batch_size
                                    ]
                                )
                            micro_batch_ids_cpu = torch.Tensor(
                                micro_batch_ids_list
                            ).int()
                            loss_scaling_cpu = torch.tensor(
                                [
                                    [
                                        1
                                        / num_mini_batch
                                        / self.config.policy.parallelism.pp_micro_batch_size
                                    ]
                                ]
                                * mini_batch_size,
                                dtype=torch.float32,
                            )

                            pp_first_stage = self.parallel_dims.pp_coord[0] == 0
                            # Pipeline Parallel forward / backward inside step() call
                            losses = [] if pp_last_stage else None
                            if pp_last_stage:
                                # Add the mini-batch ids to the input so that the last stage can know which microbatch it is processing
                                input["mini_batch_ids"] = mini_batch_ids_cpu
                                input["micro_batch_ids"] = micro_batch_ids_cpu
                                input["loss_scaling"] = loss_scaling_cpu
                            if pp_first_stage or pp_last_stage:
                                # First/Last stage: pass all inputs
                                self.pp_scheduler.step(
                                    **input,
                                    losses=losses,
                                    target=torch.empty(
                                        [mini_batch_size, 1], device=self.device
                                    ),
                                )
                            else:
                                # Middle stages: forward data from previous stage
                                self.pp_scheduler.step(position_ids=position_ids)
                            loss = (
                                torch.mean(torch.stack(losses)).to(self.device)
                                if pp_last_stage
                                else torch.tensor([-1.0], device=self.device)
                            )
                        else:
                            current_per_token_logprobs, current_advantages = (
                                self._get_per_token_logps(
                                    input, full_logits=self.model(**input)
                                )
                            )
                            if self.old_per_token_logps[local_mini_step] is None:
                                assert (
                                    i_mu == 0
                                ), "Only first iteration should append `old_per_token_logps`"
                                self.old_per_token_logps[local_mini_step] = (
                                    current_per_token_logprobs.detach()
                                )
                            else:
                                assert (
                                    i_mu > 0
                                ), "Only inner iteration should reuse `old_per_token_logps`"

                            loss, kl_loss = compute_loss(
                                current_per_token_logprobs,
                                self.old_per_token_logps[local_mini_step],
                                input.get("ref_per_token_logps", None),
                                current_advantages,
                                self.config,
                            )
                            if num_mini_batch > 1:
                                loss /= num_mini_batch
                                kl_loss /= num_mini_batch
                            loss.backward()
                    logger.debug(
                        f"[Policy] Minibatch backward step {self.mini_step + 1} at train step {self.train_step + 1} finished."
                    )
                    self.mini_step += 1
                    local_mini_step += 1
                self.execute_all_reduce()
        self.old_per_token_logps = []

        self.train_step += 1
        end_time = time.time()

        # checkpointing
        if (
            self.config.train.ckpt.enable_checkpoint
            and self.train_step % self.config.train.ckpt.save_freq == 0
            and self.train_step > 0
        ):
            logger.info(
                f"[Policy] Saving huggingface checkpoint at step {self.train_step} to {self.config.train.output_dir}..."
            )
            self.export_safetensors(
                os.path.join(
                    self.config.train.output_dir,
                    "safetensors",
                    f"step_{self.train_step}",
                ),
                trainable_only=False,
            )
            logger.info(
                f"[Policy] Saving cosmos checkpoint at step {self.train_step}..."
            )
            self.ckpt_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizers,
                scheduler=self.lr_schedulers,
                step=self.train_step,
            )
            self.save_manager.save_check(step=self.train_step)

        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            loss = loss.detach()
            global_avg_loss, global_max_loss = (  # noqa: F841
                dist_util.dist_mean(loss, self.parallel_dims.mesh["dp_cp"]),
                dist_util.dist_max(loss, self.parallel_dims.mesh["dp_cp"]),
            )
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss, global_max_kl_loss = (  # noqa: F841
                    dist_util.dist_mean(kl_loss, self.parallel_dims.mesh["dp_cp"]),
                    dist_util.dist_max(kl_loss, self.parallel_dims.mesh["dp_cp"]),
                )
        else:
            global_avg_loss = global_max_loss = loss.item()  # noqa: F841
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss = global_max_kl_loss = kl_loss.item()  # noqa: F841

        step_logging = True
        if self.config.logging.enable_logging:
            step_logging = is_wandb_available()
            if self.global_rank == 0:
                avg_rollout_len = np.mean(
                    [len(rollout["completion"]) for rollout in rollouts]
                )
                max_rollout_len = np.max(
                    [len(rollout["completion"]) for rollout in rollouts]
                )
                avg_reward = np.mean([rollout["reward"] for rollout in rollouts])
                std_reward = np.std([rollout["reward"] for rollout in rollouts])
                iter_time = end_time - start_time

                report_data = {
                    "train/loss_avg": global_avg_loss,
                    "train/loss_max": global_max_loss,
                    "train/learning_rate": self.lr_schedulers.get_last_lr()[0],
                    # TODO(dinghaoy): shall we collect the length in token
                    "train/completion_length_avg": avg_rollout_len,
                    "train/completion_length_max": max_rollout_len,
                    "train/reward_avg": avg_reward,
                    "train/reward_std": std_reward,
                    "train/iteration_time": iter_time,
                }
                if self.config.train.train_policy.kl_beta != 0.0:
                    report_data["train/kl_loss_avg"] = global_avg_kl_loss
                    report_data["train/kl_loss_max"] = global_max_kl_loss

                # FIXME(dinghaoy): only compute MFU of rank 0, if enable tp or pp,
                # it will be inaccurate. Need a reduce for all the metrics.
                if self.config.logging.report_mfu:
                    mfu = compute_mfu(
                        model=self.model,
                        inputs=inputs,
                        iter_time=iter_time,
                        num_gpus=self.world_size,
                        dtype=self.config.train.param_dtype,
                    )
                    for k, v in mfu.items():
                        report_data[f"train/{k}"] = v
                if is_wandb_available():
                    log_wandb(
                        data=report_data,
                        step=self.train_step,
                    )
                else:
                    logger.info(
                        f"Step: {self.train_step}, Loss: {global_avg_loss:.5f}, Learning rate: {self.lr_schedulers.get_last_lr()[0]:.5e}, Iteration time: {iter_time:.3f}s, Completion length: {avg_rollout_len:.0f}, Reward: {avg_reward}"
                    )

        if step_logging:
            logger.info(f"Step: {self.train_step}, Loss: {global_avg_loss:.5f}")

    @property
    def pp_loss_fn(self):
        def fake_compute_loss(
            loss: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            """
            loss: the loss of shape `[n_tokens]`
            """
            return loss.mean()

        return fake_compute_loss


def _swizzle_pp_grpo_forward(
    trainer: GRPOTrainer, ori_forward: Callable, config: CosmosConfig, *args, **kwargs
):
    args = args[1:]  # Skip self
    """
    Swizzle the forward function (only to last stage) to return the loss directly.
    """
    # [mini_batch_size]: the mini-batch index of the sample with respect to the whole batch
    # [micro_batch_size]: the micro-batch index of the sample with respect to the mini-batch
    mini_batch_ids = kwargs.pop("mini_batch_ids")
    micro_batch_ids = kwargs.pop("micro_batch_ids")
    loss_scaling = kwargs.pop("loss_scaling")
    logprobs_mask = kwargs.pop("logprob_masks")
    token_masks = kwargs.pop("token_masks")
    advantages = kwargs.pop("advantages")
    full_token_ids = kwargs.pop("input_ids")

    micro_batch_id = micro_batch_ids[0].item()
    mini_batch_id = mini_batch_ids[0].item()
    loss_scaling = loss_scaling[0].item()

    assert torch.all(
        micro_batch_ids == micro_batch_id
    ), f"micro_batch_ids are not all the same: {micro_batch_ids}"
    assert torch.all(
        mini_batch_ids == mini_batch_id
    ), f"mini_batch_ids are not all the same: {mini_batch_ids}"
    del micro_batch_ids, mini_batch_ids

    ref_per_token_logprobs = kwargs.get("ref_per_token_logps", None)
    assert (
        logprobs_mask.ndim == 2
    ), f"logprobs_mask.ndim: {logprobs_mask.ndim}, while it should be 2"
    assert (
        token_masks.ndim == 2
    ), f"token_masks.ndim: {token_masks.ndim}, while it should be 2"
    assert (
        logprobs_mask.shape == token_masks.shape
    ), f"logprobs_mask.shape: {logprobs_mask.shape}, while it should be {token_masks.shape}"
    assert (
        full_token_ids.shape[0] == logprobs_mask.shape[0]
    ), f"full_token_ids.shape[0]: {full_token_ids.shape[0]}, while it should be {logprobs_mask.shape[0]}"
    assert (
        full_token_ids.shape[0] == advantages.shape[0]
    ), f"full_token_ids.shape[0]: {full_token_ids.shape[0]}, while it should be {advantages.shape[0]}"

    # [n_tokens, n_vocab]
    current_per_token_logprobs, current_advantages = trainer._get_per_token_logps(
        input={
            "input_ids": full_token_ids,
            "logprob_masks": logprobs_mask,
            "token_masks": token_masks,
            "advantages": advantages,
        },
        full_logits=ori_forward(*args, **kwargs),
    )

    if (
        trainer.old_per_token_logps[mini_batch_id] is not None
        and len(trainer.old_per_token_logps[mini_batch_id]) > micro_batch_id
    ):
        old_per_token_logprobs = trainer.old_per_token_logps[mini_batch_id][
            micro_batch_id
        ]
        assert isinstance(old_per_token_logprobs, torch.Tensor)
        assert (
            old_per_token_logprobs.ndim == 1
        ), f"old_per_token_logprobs.ndim: {old_per_token_logprobs.ndim}, while it should be 1"
        assert (
            old_per_token_logprobs.shape == current_per_token_logprobs.shape
        ), f"old_per_token_logprobs.shape: {old_per_token_logprobs.shape}, while it should be {current_per_token_logprobs.shape}"
    else:
        old_per_token_logprobs = current_per_token_logprobs.detach()
        # Following should only happen in the first iteration
        if micro_batch_id == 0:
            # assert trainer.old_per_token_logps[mini_batch_id] is None, f"old_per_token_logps[mini_batch_id] should be None"
            # Due to the PP warmup, the first micro-batch could get processed multiple times
            trainer.old_per_token_logps[mini_batch_id] = [old_per_token_logprobs]
        else:
            assert isinstance(trainer.old_per_token_logps[mini_batch_id], list)
            trainer.old_per_token_logps[mini_batch_id].append(old_per_token_logprobs)

    # print(f"old_per_token_logprobs: {old_per_token_logprobs.shape if old_per_token_logprobs is not None else None}, {current_per_token_logprobs.shape}")

    if ref_per_token_logprobs is not None:
        assert (
            ref_per_token_logprobs.ndim == 1
        ), f"ref_per_token_logprobs.ndim: {ref_per_token_logprobs.ndim}, while it should be 1"
        assert (
            ref_per_token_logprobs.shape == current_per_token_logprobs.shape
        ), f"ref_per_token_logprobs.shape: {ref_per_token_logprobs.shape}, while it should be {current_per_token_logprobs.shape}"

    loss, _ = compute_loss(
        current_per_token_logprobs,
        old_per_token_logprobs,
        ref_per_token_logprobs,
        current_advantages,
        config,
    )

    return loss.unsqueeze(0) * loss_scaling
