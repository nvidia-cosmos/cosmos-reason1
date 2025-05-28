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
import cosmos_reason1._cpp as cosmos_c
from cosmos_reason1.utils.distributed import broadcast_object_cpu

os.environ["NCCL_CUMEM_ENABLE"] = "0"


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="cuda:nccl,cpu:gloo")

    print(f"Rank {global_rank} initialized")

    nccl_uid = broadcast_object_cpu(
        cosmos_c.create_nccl_uid() if global_rank == 0 else None
    )
    comm = cosmos_c.create_nccl_comm(nccl_uid, global_rank, world_size)
    print(f"Rank {global_rank} created comm {comm}")

    stream = torch.cuda.Stream()
    # 0 -> 4
    # 2 -> 4
    with torch.cuda.stream(stream):
        if global_rank in [0, 2]:
            send_tensor = torch.randn([16008, 5120], dtype=torch.float16).to(device)
            cosmos_c.nccl_send(send_tensor, 4, comm)
            cosmos_c.nccl_send(send_tensor, 4, comm)
        elif global_rank == 4:
            recv_tensor = torch.empty([16008, 5120], dtype=torch.float16, device=device)
            cosmos_c.nccl_recv(recv_tensor, 0, comm)
            cosmos_c.nccl_recv(recv_tensor, 2, comm)
            cosmos_c.nccl_recv(recv_tensor, 0, comm)
            cosmos_c.nccl_recv(recv_tensor, 2, comm)
    torch.cuda.synchronize()
    dist.barrier()


if __name__ == "__main__":
    main()
