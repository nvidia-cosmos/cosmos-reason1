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

from cosmos_reason1.rollout import RolloutWorkerBase
from cosmos_reason1.utils.parallelism import ParallelDims
from cosmos_reason1.policy.config import Config as CosmosConfig
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.constant import (
    COSMOS_WEIGHT_SYNC_CHECK,
    COSMOS_ROLLOUT_STEP_INTERVAL,
    COSMOS_ROLLOUT_PROMPT_QUEUE_MAX_SIZE,
)
from cosmos_reason1.utils.util import list_to_b64, b64_to_list
import torch
from transformers import AutoTokenizer, AutoConfig
from cosmos_reason1.rollout.vllm_rollout.vllm_rollout import vLLMRollout
from cosmos_reason1.dispatcher.protocol import RolloutRequest
from cosmos_reason1.dispatcher.command import (
    BuildMeshCommand,
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
    Command,
    CommandType,
)
import cosmos_reason1._cpp as _C
import requests
import torch.distributed as dist
import threading
import time
from queue import Queue
import atexit
import types
from typing import List, Tuple
from cosmos_reason1.utils.parallelism_map import (
    ParallelTopoMapperGroup,
)
from cosmos_reason1.rollout.weight_mapper import get_weight_mapper

"""
Keep in mind that torch distributed is not thread safe. So try to keep the usage in the same thread.
"""


def _patch_vllm_rollout_locked_step(rollout, consume_command):
    llm_engine = rollout.rollout_engine.llm_engine
    orig_step = llm_engine.step

    def step(self, *args, **kwargs):
        if not hasattr(self, "_cosmos_step_counter"):
            self._cosmos_step_counter = 0
        self._cosmos_step_counter += 1
        if self._cosmos_step_counter % COSMOS_ROLLOUT_STEP_INTERVAL == 0:
            consume_command()
        return orig_step(*args, **kwargs)

    llm_engine.step = types.MethodType(step, llm_engine)


class vLLMRolloutWorker(RolloutWorkerBase):
    """
    vLLMRolloutWorker will be a replica instance of single DP.
    vLLMRolloutWorker should support scaling launch.
    """

    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims) -> None:
        super(vLLMRolloutWorker, self).__init__(config, parallel_dims)
        self.cosmos_config = config
        if self.cosmos_config.rollout.parallelism.dp_shard_size == -1:
            self.cosmos_config.rollout.parallelism.dp_shard_size = (
                parallel_dims.dp_shard
            )
        assert (
            self.cosmos_config.rollout.parallelism.dp_shard_size
            == parallel_dims.dp_shard
        )
        assert (
            self.cosmos_config.rollout.parallelism.dp_shard_size > 0
        ), "[Rollout] dp_shard_size should be greater than 0."
        self.tokenizer = AutoTokenizer.from_pretrained(config.policy.model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # CommandQueue queried from controller.
        self._command_queue: Queue[Command] = Queue()
        self._prompt_queue: Queue[List[List[int, str]]] = Queue(
            maxsize=COSMOS_ROLLOUT_PROMPT_QUEUE_MAX_SIZE
        )

        self.seed = self.cosmos_config.rollout.seed

        self.rollout = vLLMRollout(
            self.cosmos_config,
            tokenizer=self.tokenizer,
            hf_config_path=self.cosmos_config.policy.model_name_or_path,
            seed=self.seed,
        )

        _patch_vllm_rollout_locked_step(self.rollout, self.consume_command)

        # communicator index for the cached communicators in C++ binding.
        self.global_commnicator_idex = -1
        # rank in current rollout replicas.
        self.rank_in_rollout_repicas = -1

        # cache for NCCL communicators for P2R.
        self.policy_to_rollout_nccl_communicators = {}

        # event reserved for shutdown.
        self.shutdown_background_task_event = threading.Event()

        self.batch_size = self.cosmos_config.rollout.batch_size

        self.background_thread: threading.Thread | None = None

        # For Polocy to Rollout weight mapping
        self.parallel_mapper = None
        hf_config = AutoConfig.from_pretrained(
            self.cosmos_config.policy.model_name_or_path
        )
        model_type = hf_config.model_type
        weight_mapper_cls = get_weight_mapper(model_type)
        self.weight_mapper = weight_mapper_cls(
            self.cosmos_config.policy.model_name_or_path
        )
        self.model_type = model_type
        self.model_config = hf_config

        self.weight_synced_event = threading.Event()

        atexit.register(self.handle_shutdown)

        # Must init the network to the controller at the end.
        self.init_comm()
        self.init_redis()
        self.prompt_end_event = threading.Event()
        self.heartbeat_thread = self.start_heartbeat(
            self.shutdown_background_task_event
        )

        self.inference_stream = torch.cuda.Stream()
        self.weight_sync_stream = torch.cuda.Stream()

    def handle_shutdown(self):
        if not self.shutdown_background_task_event.is_set():
            self.shutdown_background_task_event.set()
        if self.global_rank == 0:
            if self.background_thread is not None:
                self.background_thread.join()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join()

    def get_underlying_model(self):
        """
        Get the underlying parallelized model in vLLM internal.
        """
        return self.rollout.get_underlying_model()

    def build_global_mesh(self, build_mesh_command: BuildMeshCommand):
        logger.info(f"[Rollout] Building global mesh for {self.replica_name}")

        replica_name_to_rank = build_mesh_command.replica_name_to_rank
        if self.replica_name not in replica_name_to_rank:
            raise RuntimeError(
                f"[Rollout] Replica {self.replica_name} not found in registered replicas."
            )
        self.rank_in_rollout_repicas = replica_name_to_rank[self.replica_name]

        if len(replica_name_to_rank) == 1:
            # only one rollout replica now, no need to build mesh.
            return
        # generate key for storing the NCCL group id.
        # group_0: [rank 0 in replica 0, rank 0 in replica 1, ..., rank 0 in replica n-1]
        # group_1: [rank 1 in replica 0, rank 1 in replica 1, ..., rank 1 in replica n-1]
        # ...
        # group_m-1: [rank m-1 in replica 0, rank m-1 in replica 1, ..., rank m-1 in replica n-1]
        unique_rollout_group_key = self.get_group_unique_key(replica_name_to_rank)
        nccl_group_id = None
        max_retry = 3
        if self.rank_in_rollout_repicas == 0:
            # only replica_rank == 0 have the right to generate nccl id.
            nccl_group_id = _C.create_nccl_uid()
            base64_nccl_group_id = list_to_b64(nccl_group_id)
            while max_retry > 0:
                try:
                    r = requests.post(
                        f"{self.remote_host}/api/nccl/comm_initiator",
                        json={
                            "unique_pair_name": unique_rollout_group_key,
                            "handle_base64": base64_nccl_group_id,
                        },
                    )
                    if r.status_code != 200:
                        logger.error(
                            f"[Rollout] Error response in post nccl group_id to controller: {r.json()}"
                        )
                    else:
                        break
                except Exception as e:
                    # just logging the error now.
                    logger.error(
                        f"[Rollout] Failed in post nccl group_id to controller after {max_retry} retries: {str(e)}"
                    )
                finally:
                    max_retry -= 1

                    if max_retry == 0:
                        raise RuntimeError(
                            f"[Rollout] Failed in post nccl group_id to controller after {max_retry} retries."
                        )
                    time.sleep(0.5)

        if self.rank_in_rollout_repicas != 0:
            # other replicas should query the nccl group id from controller
            # all ranks need to wait for the rollout replica 0 finished the group_id post
            # and then they can get the group_id from controller
            # all ranks not zero in replica 0 or all ranks of other replicas need to query the group_id from controller
            nccl_group_id = self.query_nccl_unique_id_from_controller(
                unique_rollout_group_key
            )
            if nccl_group_id is None:
                raise RuntimeError(
                    "[Rollout] Failed to query nccl group_id from controller!"
                )

        # update the cached communicator index
        self.global_commnicator_idex = _C.create_nccl_comm(
            nccl_group_id, self.rank_in_rollout_repicas, len(replica_name_to_rank)
        )
        # update the replcia_name to rank dict
        self.replica_name_to_rank = replica_name_to_rank

    def query_nccl_unique_id_from_controller(self, unique_id_key: str):
        # We don't have something like dist.barrier(), so just use while True loop to query it like synchronize.
        max_retry = 1000
        nccl_group_id = None
        # all ranks not zero in replica 0 or all ranks of other replicas need to query the group_id from controller
        while max_retry > 0:
            try:
                r = requests.post(
                    f"{self.remote_host}/api/nccl/comm_acceptor",
                    json={"unique_pair_name": unique_id_key},
                )
                if r.status_code != 200:
                    logger.warning(
                        f"[Rollout] Failed in query nccl group_id from controller: {r.json()}"
                    )
                    pass
                else:
                    base64_nccl_group_id = r.json()["handle_base64"]
                    nccl_group_id = b64_to_list(base64_nccl_group_id)
                    break

            except Exception as e:
                logger.error(
                    f"[Rollout] Failed in query nccl group_id from controller: {str(e)}"
                )
            finally:
                max_retry -= 1
                if max_retry == 0:
                    raise RuntimeError(
                        f"[Rollout] Failed in query nccl group_id from controller after {max_retry} retries."
                    )
                time.sleep(0.5)
        return nccl_group_id

    @torch.no_grad()
    def sync_weight_from_policy(self, command: PolicyToRolloutUnicastCommand):
        """
        Sync the weight from policy to rollout.
        This is Policy -> Rollout replica. Will only happen between
        a pair of policy and rollout replica.
        """
        if command.dst_replica_name != self.replica_name:
            return
        rank_in_p2r = self.global_rank + command.src_replica_size
        if self.parallel_mapper is None:
            self.parallel_mapper = ParallelTopoMapperGroup(
                self.cosmos_config.policy.parallelism,
                self.cosmos_config.rollout.parallelism,
                command.src_replica_size,
                self.world_size,
                self.model_config,
                self.cosmos_config.policy.model_name_or_path,
            )

        # get the nccl_unique_id from the controller
        nccl_unique_id_key = command.src_replica_name + "_" + command.dst_replica_name
        communicator_index = -1
        if nccl_unique_id_key in self.policy_to_rollout_nccl_communicators:
            communicator_index = self.policy_to_rollout_nccl_communicators[
                nccl_unique_id_key
            ]
        else:
            nccl_group_id = self.query_nccl_unique_id_from_controller(
                nccl_unique_id_key
            )
            if nccl_group_id is None:
                raise RuntimeError(
                    "[Rollout] Failed to query nccl group_id from controller!"
                )
            communicator_index = _C.create_nccl_comm(
                nccl_group_id, rank_in_p2r, command.src_replica_size + self.world_size
            )
            self.policy_to_rollout_nccl_communicators[nccl_unique_id_key] = (
                communicator_index
            )

        with torch.cuda.stream(self.weight_sync_stream):
            # recv the weight from policy
            _, compatible_list = self.weight_mapper.generate_compatible_map(
                self.get_underlying_model()
            )
            # same as policy
            compatible_list.sort(key=lambda x: x[0])

            insts = self.parallel_mapper.generate_rollout_from_policy_insts(
                compatible_list, self.global_rank
            )
            st = time.time()

            total_bytes_received = 0
            for inst in insts:
                total_bytes_received += self.weight_mapper.recv_weight_shard(
                    self.global_rank,
                    inst,
                    communicator_index,
                )
            time_eclapsed = time.time() - st

            logger.debug(
                f"[Rolout] All {len(insts)} recv operations finished in {time_eclapsed:.3f} seconds with {total_bytes_received / (1024 * 1024)} MB received."
            )

            if not self.weight_synced_event.is_set():
                # this means we have not synced the weight yet, so we need to sync the weight.
                self.weight_sync_stream.synchronize()

            self.weight_synced_event.set()

    def broadcast_to_all_rollout_replica(
        self, broadcast_command: RolloutToRolloutBroadcastCommand
    ) -> None:
        """
        Broadcast the weight to all other rollout replicas.
        Will only happen between Rollout Replica 0 and all other Rollout Replicas.
        """
        src_replica_name: str = broadcast_command.src_replica_name
        dst_replica_names: List[str] = broadcast_command.dst_replica_names

        if len(dst_replica_names) == 1:
            # only one rollout replica, no need to broadcast.
            return

        with torch.cuda.stream(self.weight_sync_stream):
            assert (
                self.rank_in_rollout_repicas >= 0
            ), "[Rollout] rank in rollout replicas should be set before broadcast."
            assert (
                len(dst_replica_names) == len(self.replica_name_to_rank)
            ), "[Rollout] The vaild dst replicas num should match the replicas num that this worker holds."

            src_rank = self.replica_name_to_rank[src_replica_name]

            # use default stream to broadcast the weight.
            cnt = 0
            for parameter in self.get_underlying_model().parameters():
                _C.nccl_broadcast(parameter, src_rank, self.global_commnicator_idex)
                cnt += 1

            # below are just tests that broadcast takes effect.
            # if self.rank_in_rollout_repicas == src_rank:
            #     for parameter in self.get_underlying_model().parameters():
            #         with torch.no_grad():
            #             parameter.add_(1)
            #         _C.nccl_broadcast(
            #             parameter, src_rank, self.global_commnicator_idex
            #         )
            # else:
            #     for parameter in self.get_underlying_model().parameters():
            #         _C.nccl_broadcast(
            #             parameter, src_rank, self.global_commnicator_idex
            #         )
            if not self.weight_synced_event.is_set():
                # this means we have not synced the weight yet, so we need to sync the weight.
                self.weight_sync_stream.synchronize()

            self.weight_synced_event.set()

    def query_command_from_controller(self):
        """Background task to check commands from the controller"""
        while not self.shutdown_background_task_event.is_set():
            commands = []
            try:
                # blocking request
                commands = self.redis_controller.subscribe_command(self.replica_name)
            except Exception as e:
                logger.error(
                    f"[Rollout] Failed in query commands from controller for replica {self.replica_name}\n: {str(e)}"
                )
                time.sleep(0.5)  # wait and retry.
            try:
                encountered_stop = False
                for instruction in commands:
                    command = Command.depack(instruction)
                    if command.command_type == CommandType.STOP:
                        encountered_stop = True
                    logger.debug(f"[Rollout] Received command: {command.command_type}")
                    self._command_queue.put(command)
                if encountered_stop:
                    break
            except Exception as e:
                logger.error(e)
                raise e

    def generate(self) -> Tuple[List[List[str]], List[Tuple[int, str]]]:
        prompt_id_and_str_list: List[Tuple[int, str]] = self._prompt_queue.get()

        completions_per_prompt: List[List[str]] = self.rollout.rollout_generation(
            prompt_id_and_str_list=prompt_id_and_str_list,
            stream=self.inference_stream,
        )
        return completions_per_prompt, prompt_id_and_str_list

    def query_prompt(self) -> Tuple[List[Tuple[int, str]], bool]:
        assert self.global_rank == 0
        prompt_id_and_str_list = None
        is_end = False
        try:
            if not self._prompt_queue.full():
                # blocking request
                prompt_meta = requests.get(
                    f"{self.remote_host}/api/next_prompt",
                    params={"n": self.batch_size},
                )
                prompt_meta = prompt_meta.json()
                payload = prompt_meta["prompt_id_and_str_list"]
                if len(payload) > 0:
                    prompt_id_and_str_list = payload
                is_end = prompt_meta.get("is_end", is_end)
            else:
                prompt_id_and_str_list = None
        except Exception as e:
            logger.error(f"[Rollout]Failed in query prompts from controller: {str(e)}")
            prompt_id_and_str_list = None

        return prompt_id_and_str_list, is_end

    def request_new_prompts(self):
        if not self.prompt_end_event.is_set():
            prompts = [
                (None, False),
            ]
            if self.global_rank == 0:
                prompts[0] = self.query_prompt()

            if self.world_size > 1:
                dist.broadcast_object_list(
                    prompts,
                    src=0,
                    device=torch.device("cpu"),
                )
            prompts, is_end = prompts[0]
            if is_end:
                logger.info(
                    f"[Rollout] Receive prompt end, preparing exiting for {self.replica_name}."
                )
                self.prompt_end_event.set()

            if not self._prompt_queue.full() and prompts is not None:
                # if queue is full, we just abandon this prompt.
                self._prompt_queue.put(prompts)

    def consume_command(self):
        current_command = [None]
        if self.global_rank == 0:
            if not self._command_queue.empty():
                # this means we have a command to process, broadcast the command to all ranks
                current_command[0] = self._command_queue.get()

        if self.world_size > 1:
            dist.broadcast_object_list(
                current_command,
                src=0,
                device=torch.device("cpu"),
            )
            # Now all the ranks have their command broadcasted.
            dist.barrier()

        current_command = current_command[0]
        if current_command is not None:
            # this means we have a command to process, process the command
            command_type = current_command.command_type
            if command_type == CommandType.BUILD_MESH:
                self.build_global_mesh(current_command)
            elif command_type == CommandType.POLICY_TO_ROLLOUT_UNICAST:
                self.sync_weight_from_policy(current_command)
            elif command_type == CommandType.ROLLOUT_TO_ROLLOUT_BROADCAST:
                self.broadcast_to_all_rollout_replica(current_command)
            elif command_type == CommandType.STOP:
                self.shutdown_background_task_event.set()
            elif command_type == CommandType.WEIGHT_RESUME:
                raise RuntimeError(
                    "[Rollout] For rollout worker, we should not have the {CommandType.WEIGHT_RESUME} passed in."
                )

            logger.debug(
                f"[Rollout] Command executed: {current_command._serialize()} for rank: {self.global_rank}"
            )

    @torch.no_grad()
    def rollout_procedure(self):
        while not self.shutdown_background_task_event.is_set():
            self.request_new_prompts()
            self.consume_command()
            if not self.weight_synced_event.is_set():
                continue
            if self._prompt_queue.empty() or COSMOS_WEIGHT_SYNC_CHECK:
                if self.prompt_end_event.is_set() or COSMOS_WEIGHT_SYNC_CHECK:
                    # if we have no prompt and the prompt end event is set, we can stop the worker.
                    logger.info(
                        f"[Rollout] Receive prompt end event, exiting for {self.replica_name}."
                    )
                    if self.parallel_dims.tp_coord[0] == 0 and (
                        self.parallel_dims.pp_coord[0]
                        == self.parallel_dims.pp_coord[1] - 1
                    ):
                        response = RolloutRequest(
                            src_replica_name=self.replica_name,
                            prompt_idx=-1,
                            prompt="",
                            completions=[],
                            extra_info={
                                "is_end": True,
                            },
                        )
                        try:
                            logger.info(
                                f"[Rollout] Posting prompt end event to controller: {response}"
                            )
                            requests.post(
                                f"{self.remote_host}/api/rollout",
                                json=response.model_dump(),
                            )
                        except Exception as e:
                            logger.error(
                                f"[Rollout] Failed in post rollout completion to controller: {str(e)}"
                            )
                    break
                continue
            logger.debug(f"[Rollout] generate start for rank {self.global_rank}")
            completions, prompts = self.generate()
            logger.debug(f"[Rollout] generate end for rank {self.global_rank}")

            # logger.info(f"[JIAXIN] completions: {len(completions), len(completions[0]), completions[0]}")
            if self.parallel_dims.tp_coord[0] == 0 and (
                self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
            ):
                if completions is not None:
                    # only the first tp rank in the rollout replica will post the completion to the controller.
                    prompt_idxs = [prompt[0] for prompt in prompts]
                    prompt_strs = [prompt[1] for prompt in prompts]

                    response = RolloutRequest(
                        src_replica_name=self.replica_name,
                        prompt_idxs=prompt_idxs,
                        prompt_strs=prompt_strs,
                        completions=completions,
                    )

                    try:
                        requests.post(
                            f"{self.remote_host}/api/rollout",
                            json=response.model_dump(),
                        )
                    except Exception as e:
                        logger.error(
                            f"[Rollout] Failed in post rollout completion to controller: {str(e)}"
                        )
        self.weight_sync_stream.synchronize()

    def _background_task(self):
        # only rank 0 in reploca will have this background task
        self.query_command_from_controller()

    def work(self):
        # Start the thread with daemon=True, so it will exit when the main program exits.
        if self.global_rank == 0:
            # create a thread to query command as a producer
            self.background_thread = threading.Thread(
                target=self._background_task, daemon=True
            )
            self.background_thread.start()

        self.rollout_procedure()

        if self.background_thread is not None:
            self.background_thread.join()
