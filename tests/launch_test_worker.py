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
import sys
import torch
import cosmos_reason1._cpp as cosmos_c
import time
from multiprocessing import shared_memory
import numpy as np
import torch.distributed as dist
import toml


import threading
from cosmos_reason1.policy.trainer.grpo_trainer import GRPOTrainer
from cosmos_reason1.policy.trainer import Trainer
from cosmos_reason1.policy.trainer.sft_trainer import SFTTrainer
from cosmos_reason1.rollout.vllm_rollout.vllm_rollout_worker import vLLMRolloutWorker
from cosmos_reason1.rollout.vllm_rollout.vllm_rollout import vLLMRollout
import types
from cosmos_reason1.dispatcher.command import (
    PolicyToRolloutUnicastCommand,
    PolicyToPolicyUnicastCommand,
    PolicyToPolicyBroadcastCommand,
)
from cosmos_reason1.utils.parallelism_map import ParallelTopoMapperGroup
from cosmos_reason1.utils.parallelism import ParallelismConfig, ParallelDims
from cosmos_reason1.utils.distributed import (
    init_distributed,
    destroy_distributed,
)
from cosmos_reason1.dispatcher.protocol import Role
from cosmos_reason1.policy.model.gpt.weight_converter import convert_weight_from_hf
from cosmos_reason1.policy.config import Config as CosmosConfig
from cosmos_reason1.comm.base import CommMixin
from cosmos_reason1.utils.distributed import HighAvailabilitylNccl

POLICY_WORLD_SIZE = 4
ROLLOUT_WORLD_SIZE = 4


class TestModel:
    model_type = "qwen2"
    model_path = "Qwen/Qwen2.5-3B-Instruct"
    num_hidden_layers = 16

    def __init__(self, device, parallel_dims):
        self.sorted_params = [
            ("model.layers.9.input_layernorm.weight", torch.Size([1024])),
            ("model.layers.9.mlp.down_proj.weight", torch.Size([1024, 11008])),
            ("model.layers.9.mlp.gate_proj.weight", torch.Size([5504, 2048])),
            ("model.layers.9.mlp.up_proj.weight", torch.Size([5504, 2048])),
            ("model.layers.9.post_attention_layernorm.weight", torch.Size([1024])),
            ("model.layers.9.self_attn.k_proj.bias", torch.Size([128])),
            ("model.layers.9.self_attn.k_proj.weight", torch.Size([128, 2048])),
            ("model.layers.9.self_attn.o_proj.weight", torch.Size([1024, 2048])),
            ("model.layers.9.self_attn.q_proj.bias", torch.Size([1024])),
            ("model.layers.9.self_attn.q_proj.weight", torch.Size([1024, 2048])),
            ("model.layers.9.self_attn.v_proj.bias", torch.Size([128])),
            ("model.layers.9.self_attn.v_proj.weight", torch.Size([128, 2048])),
            ("lm_head.weight", torch.Size([75968, 2048])),
            ("model.norm.weight", torch.Size([1024])),
            ("model.embed_tokens.weight", torch.Size([75968, 2048])),
        ]
        self.sorted_params.sort(key=lambda x: x[0])
        self.device = device
        self.parallel_dims = parallel_dims
        self.tensors = [
            (
                k,
                torch.arange(v.numel(), dtype=torch.float32, device=self.device)
                .reshape(v)
                .to(self.device)
                * 0.001,
            )
            for k, v in self.sorted_params
        ]
        self.sharded_tensors = {}
        for k, v in self.tensors:
            self.sharded_tensors[k] = convert_weight_from_hf(
                v, k, self.model_type, self.parallel_dims
            )[1]
        self.sorted_sharded_params = [
            (k, self.sharded_tensors[k].shape) for k, _ in self.sorted_params
        ]


class TestPolicy:
    def __init__(self, name, policy_world_size, rollout_world_size, rollouts_comm):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.global_rank = int(os.environ.get("RANK", 0))
        self.role = Role.POLICY
        self.world_size = policy_world_size
        policy_parallelism_config = ParallelismConfig(
            dp_shard_size=2, cp_size=1, tp_size=2, pp_size=1
        )
        rollout_parallelism_config = ParallelismConfig(
            dp_shard_size=1, cp_size=1, tp_size=4, pp_size=1
        )
        self.parallel_dims = ParallelDims.from_config(
            parallesim_config=policy_parallelism_config,
        )
        self.parallel_dims.build_mesh(device_type="cuda")
        self.model = TestModel(self.device, self.parallel_dims)
        self.parallel_mapper = ParallelTopoMapperGroup(
            policy_parallelism_config,
            rollout_parallelism_config,
            policy_world_size,
            rollout_world_size,
            self.model,
            self.model.model_path,
        )
        self.replica_name = name
        self.rollouts_comm = rollouts_comm
        self.policy_to_rollout_insts = None
        self.map_w_from_policy_to_rollout = self.model.sharded_tensors
        self.model.sorted_params = self.model.sorted_sharded_params
        self.p2r_nccl_uuids = rollouts_comm
        self.train_stream = torch.cuda.Stream()

    def execute_policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        pass

    def precollect_parameters_for_sync(self):
        return {}


class TestRollout:
    def __init__(self, name, policy_world_size, rollout_world_size, policies_comm):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.global_rank = int(os.environ.get("RANK", 0))
        self.role = Role.ROLLOUT
        self.world_size = rollout_world_size
        self.policy_to_rollout_nccl_communicators = policies_comm
        policy_parallelism_config = ParallelismConfig(
            dp_shard_size=2, cp_size=1, tp_size=2, pp_size=1
        )
        rollout_parallelism_config = ParallelismConfig(
            dp_shard_size=1, cp_size=1, tp_size=4, pp_size=1
        )
        self.replica_name = name
        self.parallel_dims = ParallelDims.from_config(
            rollout_parallelism_config,
        )
        self.parallel_dims.build_mesh(device_type="cuda")
        self.model = TestModel(self.device, self.parallel_dims)
        self.parallel_mapper = ParallelTopoMapperGroup(
            policy_parallelism_config,
            rollout_parallelism_config,
            policy_world_size,
            rollout_world_size,
            self.model,
            self.model.model_path,
        )
        self.weight_mapper = self.parallel_mapper.weight_mapper
        compatibale_map = self.model.sharded_tensors
        compatibale_list = self.model.sorted_sharded_params
        operate_compatibale_map = {
            k: torch.zeros(v.shape, dtype=v.dtype).to(self.device)
            for k, v in compatibale_map.items()
        }
        self.ref_compatibale_map = compatibale_map

        def custom_generate_compatible_map(self, model):
            self.compatible_weight_map = compatibale_map
            self.compatible_list = compatibale_list
            return operate_compatibale_map, compatibale_list

        self.operate_compatibale_map = operate_compatibale_map
        self.weight_mapper.generate_compatible_map = types.MethodType(
            custom_generate_compatible_map, self.weight_mapper
        )
        self.weight_synced_event = threading.Event()
        self.inference_stream = torch.cuda.Stream()

    def get_underlying_model(self):
        return None

    def sync_weight_from_policy(self, command: PolicyToRolloutUnicastCommand):
        pass


def run_policy_send_to_rollout(shm_name, shm_size, rank):
    """Run as a test policy process to send to rollout process"""
    # Set up NCCL communicator
    policy_name = "policy"
    rollout_name = "rollout"

    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)

    command = PolicyToRolloutUnicastCommand(
        policy_name, rollout_name, POLICY_WORLD_SIZE, ROLLOUT_WORLD_SIZE, ""
    )

    try:
        if rank == 0:
            nccl_uid = cosmos_c.create_nccl_uid()
            # Create shared memory for NCCL UID
            uid_array = np.ndarray((shm_size + 1,), dtype=np.int64, buffer=shm.buf)
            # Copy NCCL UID to shared memory
            uid_array[:-1] = nccl_uid
            uid_array[-1] = 1
        else:
            uid_array = np.ndarray((shm_size + 1,), dtype=np.int64, buffer=shm.buf)
            while uid_array[-1] == 0:
                time.sleep(0.001)
            assert uid_array[-1] == 1, "Sender process did not set UID correctly"
            nccl_uid = uid_array[:-1].tolist()

        # Create NCCL communicator after UID is shared
        comm_idx = cosmos_c.create_nccl_comm(
            nccl_uid, rank, POLICY_WORLD_SIZE + ROLLOUT_WORLD_SIZE
        )
        policy = TestPolicy(
            policy_name,
            POLICY_WORLD_SIZE,
            ROLLOUT_WORLD_SIZE,
            {policy_name + "_" + rollout_name: comm_idx},
        )
        policy.execute_policy_to_rollout_unicast = types.MethodType(
            GRPOTrainer.execute_policy_to_rollout_unicast, policy
        )
        policy.execute_policy_to_rollout_unicast(command)
        policy.train_stream.synchronize()

    finally:
        # Detach from shared memory
        shm.close()


def run_rollout_recv_from_policy(shm_name, shm_size, rank):
    """Run as a rollout process to receive from policy process"""
    # Set up NCCL communicator
    policy_name = "policy"
    rollout_name = "rollout"

    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)

    command = PolicyToRolloutUnicastCommand(
        policy_name, rollout_name, POLICY_WORLD_SIZE, ROLLOUT_WORLD_SIZE, ""
    )
    try:
        # Get NCCL UID from shared memory
        uid_array = np.ndarray((shm_size + 1,), dtype=np.int64, buffer=shm.buf)
        while uid_array[-1] == 0:
            time.sleep(0.001)
        assert uid_array[-1] == 1, "Sender process did not set UID correctly"
        nccl_uid = uid_array[:-1].tolist()
        # Create NCCL communicator with shared UID
        comm_idx = cosmos_c.create_nccl_comm(
            nccl_uid, rank + POLICY_WORLD_SIZE, POLICY_WORLD_SIZE + ROLLOUT_WORLD_SIZE
        )

        rollout = TestRollout(
            rollout_name,
            POLICY_WORLD_SIZE,
            ROLLOUT_WORLD_SIZE,
            {policy_name + "_" + rollout_name: comm_idx},
        )
        rollout.sync_weight_from_policy = types.MethodType(
            vLLMRolloutWorker.sync_weight_from_policy, rollout
        )
        rollout.sync_weight_from_policy(command)
        rollout.inference_stream.synchronize()

        for k, v in rollout.operate_compatibale_map.items():
            torch.allclose(v, rollout.ref_compatibale_map[k])

    finally:
        # Detach from shared memory
        shm.close()


def policy_to_policy_sync_common(
    shm_names,
    shm_size,
    rank,
    send,
    nccl_rank,
    nccl_size,
    policy_name,
    replica_name_to_rank,
    command,
):
    """Run as a policy process to perform unicast to another policy process or broadcast to all policy processes"""
    # Attach to shared memory
    shm_names = shm_names.split(",")
    shm_name = shm_names[rank]
    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        # Get NCCL UID from shared memory
        if send:
            nccl_uid = cosmos_c.create_nccl_uid()
            # Create shared memory for NCCL UID
            uid_array = np.ndarray((shm_size + 1,), dtype=np.int64, buffer=shm.buf)
            # Copy NCCL UID to shared memory
            uid_array[:-1] = nccl_uid
            uid_array[-1] = 1
        else:
            uid_array = np.ndarray((shm_size + 1,), dtype=np.int64, buffer=shm.buf)
            # Wait for sender process to set UID
            while uid_array[-1] == 0:
                time.sleep(0.001)
            assert uid_array[-1] == 1, "Sender process did not set UID correctly"
            nccl_uid = uid_array[:-1].tolist()

        # Create NCCL communicator with shared UID
        comm_idx = cosmos_c.create_nccl_comm(nccl_uid, nccl_rank, nccl_size)

        # Construct the model and trainer
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(
            os.path.dirname(cur_dir),
            "configs",
            "qwen2-5",
            "qwen2-5-3b-p-fsdp1-tp2-r-tp2-pp1-grpo.toml",
        )

        with open(config_path, "r") as f:
            config_dict = toml.load(f)

        cosmos_config = CosmosConfig.from_dict(
            config_dict,
        )
        parallel_dims = ParallelDims.from_config(cosmos_config.policy.parallelism)
        parallel_dims.build_mesh(device_type="cuda")

        def dummy(self, *args, **kwargs):
            pass

        def dummy_init_nccl(self, replica_name, global_rank, controller_hosts):
            pass

        HighAvailabilitylNccl.__init__ = dummy_init_nccl

        class FakeNCCL:
            def __init__(self, comm_idx):
                self.comm_idx = comm_idx

            def get_rank(self, replica_name: str):
                if replica_name in replica_name_to_rank:
                    return replica_name_to_rank[replica_name]
                else:
                    raise ValueError(
                        f"Replica name {replica_name} not found in mapping."
                    )

            def broadcast(self, tensor: torch.Tensor, src_replica: str):
                src_rank = self.get_rank(src_replica)
                cosmos_c.nccl_broadcast(tensor, src_rank, self.comm_idx)

            def send(self, tensor: torch.Tensor, dst_replica: str):
                dst_rank = self.get_rank(dst_replica)
                cosmos_c.nccl_send(tensor, dst_rank, self.comm_idx)

            def recv(self, tensor: torch.Tensor, src_replica: str):
                src_rank = self.get_rank(src_replica)
                cosmos_c.nccl_recv(tensor, src_rank, self.comm_idx)

            def shutdown(self):
                pass

        Trainer.init_comm = dummy
        CommMixin.init_redis = dummy
        CommMixin.start_heartbeat = dummy
        CommMixin.replica_name = policy_name
        CommMixin.remote_hosts = ["localhost:0"]
        policy = GRPOTrainer(cosmos_config, parallel_dims)
        policy.model_load_from_hf()
        policy.replica_name = policy_name
        policy.inter_policy_nccl = FakeNCCL(comm_idx)
        policy.mesh_ready = True
        policy.replica_name_to_rank = replica_name_to_rank

        def sample_tensor():
            sample_tensors = []
            self_state_dict = policy.model.state_dict()
            sample_tensors.append(self_state_dict[sorted(self_state_dict.keys())[0]])
            sample_tensors.append(self_state_dict[sorted(self_state_dict.keys())[-1]])

            optimizer_state = policy.optimizers.state_dict()
            sample_tensors.append(optimizer_state[sorted(optimizer_state.keys())[0]])
            sample_tensors.append(optimizer_state[sorted(optimizer_state.keys())[-1]])

            lr_sheduler_state = policy.lr_schedulers.state_dict()
            sample_tensors.append(
                lr_sheduler_state[sorted(lr_sheduler_state.keys())[0]]
            )
            sample_tensors.append(
                lr_sheduler_state[sorted(lr_sheduler_state.keys())[-1]]
            )
            sample_tensors = [
                tensor.to_local().cpu()
                if isinstance(tensor, torch.distributed.tensor.DTensor)
                else tensor.cpu()
                if isinstance(tensor, torch.Tensor)
                else tensor
                for tensor in sample_tensors
            ]
            return sample_tensors

        if not send:
            sample_tensors = sample_tensor()

        if isinstance(command, PolicyToPolicyUnicastCommand):
            policy.execute_policy_to_policy_unicast(command)
        elif isinstance(command, PolicyToPolicyBroadcastCommand):
            policy.execute_policy_to_policy_broadcast(command)

        if not send:
            origin_sample_tensors = sample_tensors
            sample_tensors = sample_tensor()
            for tensor, origin_tensor in zip(sample_tensors, origin_sample_tensors):
                if isinstance(tensor, torch.Tensor):
                    assert torch.allclose(
                        tensor, origin_tensor
                    ), f"Tensor values do not match {tensor} {origin_tensor}"
                elif isinstance(tensor, bool):
                    assert (
                        tensor == origin_tensor
                    ), f"Tensor values do not match {tensor} {origin_tensor}"
    finally:
        # Detach from shared memory
        shm.close()


def run_policy_unicast_to_policy(shm_names, shm_size, rank, send):
    """Run as a policy process to perform unicast to another policy process"""
    policy_src_name = "policy_src"
    policy_dst_name = "policy_dst"
    command = PolicyToPolicyUnicastCommand(policy_src_name, policy_dst_name, "")
    nccl_rank = 0 if send else 1
    nccl_size = 2
    replica_name_to_rank = {policy_src_name: 0, policy_dst_name: 1}
    policy_name = policy_src_name if send else policy_dst_name
    # Call the common function to handle both send and receive
    policy_to_policy_sync_common(
        shm_names,
        shm_size,
        rank,
        send,
        nccl_rank,
        nccl_size,
        policy_name,
        replica_name_to_rank,
        command,
    )


def run_policy_broadcast_to_policy(shm_names, shm_size, rank, total_rep, self_rep):
    """Run as a policy process to perform broadcast to all policy processes"""
    policy_name = "policy_" + str(self_rep)
    policy_src = "policy_0"
    policy_dsts = ["policy_" + str(rep) for rep in range(total_rep)]
    command = PolicyToPolicyBroadcastCommand(policy_src, policy_dsts, "")
    nccl_rank = self_rep
    nccl_size = total_rep
    replica_name_to_rank = {"policy_" + str(i): i for i in range(total_rep)}
    send = policy_src == policy_name
    # Call the common function to handle both send and receive
    policy_to_policy_sync_common(
        shm_names,
        shm_size,
        rank,
        send,
        nccl_rank,
        nccl_size,
        policy_name,
        replica_name_to_rank,
        command,
    )


def run_dummy_policy():
    """Run as a dummy policy process for testing"""
    from cosmos_reason1.policy.train import run_train

    def dummy_train(self):
        pass

    def dummy_model_load_from_hf(self):
        self.model_ready = True

    def dummy_execute_policy_to_rollout_unicast(self, command):
        return True

    GRPOTrainer.train = dummy_train
    GRPOTrainer.model_load_from_hf = dummy_model_load_from_hf

    def get_policy_command_handler(cls, command_type):
        if command_type == PolicyToRolloutUnicastCommand:
            return dummy_execute_policy_to_rollout_unicast
        return cls.policy_command_handler_registry.get_command_handler(command_type)

    GRPOTrainer.get_policy_command_handler = get_policy_command_handler
    SFTTrainer.train = dummy_train
    run_train()


def run_dummy_rollout():
    """Run as a dummy rollout process for testing purposes"""
    from cosmos_reason1.rollout.rollout_entrance import run_rollout

    def dummy_sync_weight_from_policy(self, command):
        self.weight_synced_event.set()

    def generate(self):
        prompt_id_and_payload_list = self._prompt_queue.get()
        payloads = [x[1] for x in prompt_id_and_payload_list]
        completions_per_prompt = [[x] for x in payloads]
        return completions_per_prompt, prompt_id_and_payload_list

    def get_rollout_command_handler(cls, command_type):
        if command_type == PolicyToRolloutUnicastCommand:
            return dummy_sync_weight_from_policy
        return cls.rollout_command_handler_registry.get_command_handler(command_type)

    vLLMRolloutWorker.generate = generate
    vLLMRolloutWorker.get_rollout_command_handler = get_rollout_command_handler

    def dummy_init(self, config, tokenizer, **kwargs):
        class Llm_engine:
            def step(self, *args, **kwargs):
                pass

        class Rollout_engine:
            llm_engine = Llm_engine()

        self.rollout_engine = Rollout_engine()

    vLLMRollout.__init__ = dummy_init
    run_rollout()


def dummy_controller():
    """Run as a dummy controller process for testing purposes"""


def main():
    # Get shared memory name and size from command line arguments
    shm_name = sys.argv[1]
    shm_size = int(sys.argv[2])
    mode = sys.argv[3]

    if mode == "dummy_policy":
        # Dummy policy process for testing
        run_dummy_policy()
        exit(0)
    elif mode == "dummy_rollout":
        # Dummy rollout process for testing
        run_dummy_rollout()
        exit(0)

    # Initialize distributed environment
    init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank} started with mode {mode} {torch.cuda.current_device()}")

    if mode == "policy_send_to_rollout":
        assert (
            world_size == POLICY_WORLD_SIZE
        ), "World size must match POLICY_WORLD_SIZE for policy process"
        run_policy_send_to_rollout(shm_name, shm_size, rank)
    elif mode == "rollout_recv_from_policy":
        assert (
            world_size == ROLLOUT_WORLD_SIZE
        ), "World size must match ROLLOUT_WORLD_SIZE for rollout process"
        run_rollout_recv_from_policy(shm_name, shm_size, rank)
    elif mode == "policy_send_to_policy":
        run_policy_unicast_to_policy(shm_name, shm_size, rank, True)
    elif mode == "policy_recv_from_policy":
        run_policy_unicast_to_policy(shm_name, shm_size, rank, False)
    elif mode.startswith("policy_broadcast_to_policy"):
        total_rep = int(mode.split(",")[1])
        self_rep = int(mode.split(",")[2])
        run_policy_broadcast_to_policy(shm_name, shm_size, rank, total_rep, self_rep)
    else:
        raise ValueError("Invalid mode.")
    # Clean up distributed environment
    destroy_distributed()


if __name__ == "__main__":
    main()
