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
import pytest
import torch
import cosmos_reason1._cpp as cosmos_c
import subprocess
import sys
from multiprocessing import shared_memory
import numpy as np


def test_policy_to_policy_unicast():
    """Test NCCL communication between multiple ranks using torchrun."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # Create NCCL UID and shared memory
    nccl_uid = cosmos_c.create_nccl_uid()
    nccl_uid_tensor = torch.tensor(nccl_uid, dtype=torch.int64)
    shms = []
    world_size = 2
    for i in range(world_size):
        shms.append(
            shared_memory.SharedMemory(
                create=True,
                size=(nccl_uid_tensor.numel() + 1) * nccl_uid_tensor.element_size(),
            )
        )
        uid_array = np.ndarray(
            (nccl_uid_tensor.numel() + 1,), dtype=np.int64, buffer=shms[-1].buf
        )
        uid_array[-1] = 0

    try:
        # Create the Python command for torchrun
        policy_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 2 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            ",".join([shm.name for shm in shms]),
            str(nccl_uid_tensor.numel()),
            "policy_send_to_policy",
        ]
        policy_dst_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 2 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            ",".join([shm.name for shm in shms]),
            str(nccl_uid_tensor.numel()),
            "policy_recv_from_policy",
        ]
        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        # Start the process
        policy_process = subprocess.Popen(
            policy_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=policy_env,
        )
        policy_dst_env = dict(os.environ)
        policy_dst_env["CUDA_VISIBLE_DEVICES"] = "2,3"
        policy_dst_process = subprocess.Popen(
            policy_dst_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=policy_dst_env,
        )

        try:
            # Wait for process to complete
            for process in [policy_process, policy_dst_process]:
                stdout, stderr = process.communicate()

                # Check if process completed successfully
                assert process.returncode == 0, f"Process failed: {stderr.decode()}"

        finally:
            # Ensure process is terminated
            for process in [policy_process, policy_dst_process]:
                process.wait()
    finally:
        # Clean up shared memory
        try:
            for shm in shms:
                shm.close()
                shm.unlink()
        except FileNotFoundError:
            # Ignore if shared memory is already unlinked
            pass


def test_policy_to_policy_broadcast():
    """Test NCCL communication between multiple ranks using torchrun."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # Create NCCL UID and shared memory
    nccl_uid = cosmos_c.create_nccl_uid()
    nccl_uid_tensor = torch.tensor(nccl_uid, dtype=torch.int64)
    shms = []
    world_size = 2
    for i in range(world_size):
        shms.append(
            shared_memory.SharedMemory(
                create=True,
                size=(nccl_uid_tensor.numel() + 1) * nccl_uid_tensor.element_size(),
            )
        )
        uid_array = np.ndarray(
            (nccl_uid_tensor.numel() + 1,), dtype=np.int64, buffer=shms[-1].buf
        )
        uid_array[-1] = 0

    try:
        # Create the Python command for torchrun
        policy_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 2 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            ",".join([shm.name for shm in shms]),
            str(nccl_uid_tensor.numel()),
            ",".join(["policy_broadcast_to_policy", "4", "0"]),
        ]
        policy_dst_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 2 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            ",".join([shm.name for shm in shms]),
            str(nccl_uid_tensor.numel()),
            ",".join(["policy_broadcast_to_policy", "4", "0"]),
        ]
        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        # Start the process
        policy_process = subprocess.Popen(
            policy_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=policy_env,
        )
        policy_dst_env = dict(os.environ)
        policy_dst_process = []

        envs = ["2,3", "4,5", "6,7"]
        reps = ["1", "2", "3"]
        for env, rep in zip(envs, reps):
            policy_dst_env["CUDA_VISIBLE_DEVICES"] = env
            policy_dst_cmd[-1] = ",".join(["policy_broadcast_to_policy", "4", rep])
            policy_dst_process.append(
                subprocess.Popen(
                    policy_dst_cmd,
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                    env=policy_dst_env,
                )
            )
        try:
            # Wait for process to complete
            for process in [policy_process] + policy_dst_process:
                stdout, stderr = process.communicate()

                # Check if process completed successfully
                assert process.returncode == 0, f"Process failed: {stderr.decode()}"

        finally:
            # Ensure process is terminated
            for process in [policy_process] + policy_dst_process:
                process.wait()
    finally:
        # Clean up shared memory
        try:
            for shm in shms:
                shm.close()
                shm.unlink()
        except FileNotFoundError:
            # Ignore if shared memory is already unlinked
            pass


if __name__ == "__main__":
    pytest.main([__file__])
