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

from typing import List

from utils.pynccl import int

def watchdog_enter() -> None: ...
def watchdog_exit(abort: bool) -> None: ...
def create_nccl_comm(
    uid_chars: List[int],
    rank: int,
    world_size: int,
    timeout_ms: int,
) -> int: ...
def create_nccl_uid() -> List[int]: ...
def nccl_abort(comm_idx: int) -> None: ...
def get_nccl_comm_count(comm_idx: int) -> int: ...
def get_default_timeout_ms() -> int: ...
def nccl_broadcast(
    tensor: int,
    count: int,
    dtype: int,
    rank: int,
    comm_idx: int,
    stream: int,
    timeout_ms: int,
) -> None: ...
def nccl_send(
    tensor: int,
    count: int,
    dtype: int,
    peer: int,
    comm_idx: int,
    stream: int,
    timeout_ms: int,
) -> None: ...
def nccl_recv(
    tensor: int,
    count: int,
    dtype: int,
    peer: int,
    comm_idx: int,
    stream: int,
    timeout_ms: int,
) -> None: ...
def nccl_allreduce(
    sendbuff: int,
    recvbuff: int,
    count: int,
    dtype: int,
    op: int,
    comm_idx: int,
    stream: int,
    timeout_ms: int,
) -> None: ...
def nccl_alltoall(
    sendbuff: int,
    recvbuff: int,
    total_size: int,
    dtype: int,
    comm_idx: int,
    stream: int,
    timeout_ms: int,
) -> None: ...
