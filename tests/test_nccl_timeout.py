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
import torch.distributed as dist
import subprocess

# For testing, set the timeout to 60000ms
os.environ["COSMOS_NCCL_TIMEOUT_MS"] = "60000"
import cosmos_reason1._cpp as cosmos_c
from cosmos_reason1.utils.distributed import broadcast_object_cpu, nccl_timeout_watchdog


def routine(N: int, device: torch.device, rank: int, world_size: int):
    nccl_uid = broadcast_object_cpu(cosmos_c.create_nccl_uid() if rank == 0 else None)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        comm = cosmos_c.create_nccl_comm(nccl_uid, rank, world_size)
        try:
            with nccl_timeout_watchdog(wait_stream=True):
                send_buffer = torch.arange(N, dtype=torch.float32, device=device)
                recv_buffer = torch.zeros(N, dtype=torch.float32, device=device)

                if rank != 3:
                    # Simulate a failed allreduce on rank 3
                    try:
                        cosmos_c.nccl_allreduce(send_buffer, recv_buffer, 0, comm)
                        print(f"[RANK {rank}] arrived here")
                    except Exception as e:
                        print(f"[RANK {rank}] error in nccl_allreduce: {e}")
        except Exception as e:
            print(f"[RANK {rank}] error in nccl_timeout_watchdog: {e}")

        if rank < 2:
            # Test that the new group still works after the previous nccl failure
            print(f"[rank {rank}] creating 2nd comm")
            world_size = 2
            group_ranks = [0, 1]
            new_group = dist.new_group(ranks=group_ranks, backend="cuda:nccl,cpu:gloo")

            nccl_uid = broadcast_object_cpu(
                cosmos_c.create_nccl_uid() if rank == 0 else None, group=new_group
            )
            comm = cosmos_c.create_nccl_comm(nccl_uid, rank, world_size)
            send_tensor = torch.arange(N, dtype=torch.float32, device=device)
            recv_tensor = torch.zeros(N, dtype=torch.float32, device=device)
            cosmos_c.nccl_allreduce(send_tensor, recv_tensor, 0, comm)
            torch.testing.assert_close(
                recv_tensor,
                torch.arange(N, dtype=torch.float32, device=device) * world_size,
            )
        print(f"[rank {rank}] finished the test")


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="cuda:nccl,cpu:gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    ###
    # Global timeout: 60s
    # 1. We initialize comm and launch allreduce on all ranks except rank 3
    # 2. Rank 3 will sleep for 100s
    #     By design, the allreduce could trigger a cpu waiting timeout on all ranks except rank 3
    # 3. Rank 0 and 1 will continue to run and complete the allreduce
    routine(N=8, device=device, rank=rank, world_size=world_size)

    print(f"[RANK {rank}] Running 2nd routine\n")
    from importlib.metadata import version

    nccl_v = version("nvidia-nccl-cu12")
    if nccl_v >= "2.26.2":
        routine(N=512 * 4096 * 512, device=device, rank=rank, world_size=world_size)


if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK") is None:
        n_gpu = torch.cuda.device_count()
        command = [
            "torchrun",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(n_gpu),
            os.path.abspath(__file__),
        ]
        subprocess.run(command)
    else:
        main()
