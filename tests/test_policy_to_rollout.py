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
import subprocess
import sys
from multiprocessing import shared_memory
import numpy as np
from tests.launch_test_worker import POLICY_WORLD_SIZE, ROLLOUT_WORLD_SIZE
from cosmos_reason1.utils.pynccl import (
    create_nccl_uid,
)


def test_policy_to_rollout_wieght_sync():
    """Test NCCL communication between multiple ranks using torchrun."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # Create NCCL UID and shared memory
    nccl_uid = create_nccl_uid()
    nccl_uid_tensor = torch.tensor(nccl_uid, dtype=torch.int64)
    shm = shared_memory.SharedMemory(
        create=True, size=(nccl_uid_tensor.numel() + 1) * nccl_uid_tensor.element_size()
    )
    uid_array = np.ndarray(
        (nccl_uid_tensor.numel() + 1,), dtype=np.int64, buffer=shm.buf
    )
    uid_array[-1] = 0

    try:
        # Create the Python command for torchrun
        policy_cmd = [
            "torchrun",
            f"--nproc_per_node={POLICY_WORLD_SIZE}",  # Use 4 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            shm.name,
            str(nccl_uid_tensor.numel()),
            "policy_send_to_rollout",
        ]
        rollout_cmd = [
            "torchrun",
            f"--nproc_per_node={ROLLOUT_WORLD_SIZE}",  # Use 4 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            shm.name,
            str(nccl_uid_tensor.numel()),
            "rollout_recv_from_policy",
        ]
        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # Start the process
        policy_process = subprocess.Popen(
            policy_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=policy_env,
        )
        rollout_env = dict(os.environ)
        rollout_env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
        rollout_process = subprocess.Popen(
            rollout_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=rollout_env,
        )

        try:
            # Wait for process to complete
            for process in [policy_process, rollout_process]:
                stdout, stderr = process.communicate()

                # Check if process completed successfully
                assert process.returncode == 0, f"Process failed: {stderr.decode()}"

        finally:
            # Ensure process is terminated
            for process in [policy_process, rollout_process]:
                process.wait()
    finally:
        # Clean up shared memory
        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            # Ignore if shared memory is already unlinked
            pass


if __name__ == "__main__":
    pytest.main([__file__])
