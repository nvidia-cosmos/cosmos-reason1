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
import time
from contextlib import contextmanager
from importlib.metadata import version
from typing import Optional, List

import torch
from torch.distributed import ReduceOp
from torch.cuda import Stream

import cosmos_reason1._cpp as cosmos_c
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.util import do_once


class NcclDataType:
    # copy from nccl.h
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclFloat8e4m3 = 10
    ncclFloat8e5m2 = 11
    ncclNumTypes = 12

    map_dtype = {
        torch.int8: ncclInt8,
        torch.uint8: ncclUint8,
        torch.int32: ncclInt32,
        torch.uint32: ncclUint32,
        torch.int64: ncclInt64,
        torch.uint64: ncclUint64,
        torch.float16: ncclFloat16,
        torch.float16: ncclHalf,
        torch.float32: ncclFloat32,
        torch.float: ncclFloat,
        torch.float64: ncclFloat64,
        torch.double: ncclDouble,
        torch.bfloat16: ncclBfloat16,
        # torch.float8_e4m3fn: ncclFloat8e4m3, # need check
        torch.float8_e5m2: ncclFloat8e5m2,
    }

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype not in cls.map_dtype:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return int(cls.map_dtype[dtype])


class NcclRedOp:
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4

    map_redop = {
        ReduceOp.SUM: ncclSum,
        ReduceOp.PRODUCT: ncclProd,
        ReduceOp.MAX: ncclMax,
        ReduceOp.MIN: ncclMin,
        ReduceOp.AVG: ncclAvg,
    }

    @classmethod
    def to_nccl_reduce_op(cls, opType: ReduceOp.RedOpType) -> int:
        if opType not in cls.map_redop:
            raise ValueError(f"Unsupported opType: {opType}")
        return cls.map_redop[opType]


def check_tensor(tensor: torch.Tensor):
    if not tensor.is_cuda:
        raise ValueError("Tensor must be a CUDA tensor")
    if not tensor.is_contiguous:
        raise ValueError("Tensor must be contiguous")
    if tensor.numel() == 0:
        raise ValueError("Tensor must have non-zero number of elements")
    if tensor.device.index != torch.cuda.current_device():
        raise ValueError("Tensor's device does not match the current CUDA device")
    return True


def get_cuda_stream(cuda_stream: Optional[Stream] = None) -> int:
    # Be attention, when cuda not set stream, the current stream is 0 (nullptr),
    # But don't worry, the nccl api will handle this case.
    if cuda_stream is None:
        return torch.cuda.current_stream().cuda_stream
    return cuda_stream.cuda_stream


def get_tensor_ptr(tensor: torch.Tensor) -> int:
    return int(tensor.data_ptr())


def get_nccl_timeout_ms() -> int:
    timeout_ms = int(
        os.environ.get("COSMOS_NCCL_TIMEOUT_MS", cosmos_c.get_default_timeout_ms())
    )
    if timeout_ms < 0:
        raise ValueError(
            f"COSMOS_NCCL_TIMEOUT_MS must be non-negative, but got {timeout_ms}"
        )
    return timeout_ms


# below wrap torch.Tensor to pure c++ code binding


@contextmanager
def nccl_timeout_watchdog(wait_stream=False, timeout_ms: int = None):
    """
    Context manager to monitor NCCL operations and raise an error if they take longer than a specified timeout.
    Important: do not call any synchronous API:
    - torch.cuda.synchronize()
    - torch.cuda.stream.synchronize()
    - torch.cuda.stream.wait_stream()
    - torch.cuda.event.wait()
    - ...

    Args:
        wait_stream (bool): If True, wait for the NCCL operation to complete before raising an error.
        If False, just wait until all NCCL operations are enqueued to the stream.
    """
    nccl_v = version("nvidia-nccl-cu12")
    if nccl_v < "2.26.2":

        @do_once
        def warn_nccl_version():
            logger.warning(
                "NCCL version is less than 2.26.2, which is known to hang in some cases, please upgrade to a newer version"
            )

        warn_nccl_version()

    start_time = time.time()
    threshold_ms = timeout_ms if timeout_ms is not None else get_nccl_timeout_ms()
    # Enter the watchdog context
    cosmos_c.watchdog_enter()
    error_raised = False
    try:
        yield
    except Exception as e:
        error_raised = True
        raise e
    finally:
        if wait_stream and not error_raised:
            event = torch.cuda.Event()
            event.record()
            while not event.query():
                if time.time() - start_time > threshold_ms / 1000:
                    cosmos_c.watchdog_exit(abort=True)
                    raise RuntimeError(
                        f"NCCL operation took {time.time() - start_time} seconds, which is longer than the threshold {threshold_ms} ms"
                    )
        cosmos_c.watchdog_exit(abort=error_raised)


def create_nccl_comm(
    uid_chars: List[int], rank: int, world_size: int, timeout_ms: Optional[int] = None
) -> int:
    """
    Create a NCCL communication group. Return the comm_idx.
    """
    timeout_ms = get_nccl_timeout_ms() if timeout_ms is None else timeout_ms
    return cosmos_c.create_nccl_comm(uid_chars, rank, world_size, timeout_ms)


def create_nccl_uid() -> List[int]:
    """
    Create a NCCL unique identifier.
    """
    return cosmos_c.create_nccl_uid()


def nccl_abort(comm_idx: int):
    """
    Abort the NCCL communication group.
    """
    cosmos_c.nccl_abort(comm_idx)


def get_nccl_comm_nranks(comm_idx: int) -> int:
    """
    Get the number of ranks in the NCCL communication group.
    """
    return cosmos_c.get_nccl_comm_count(comm_idx)


def nccl_broadcast(
    tensor: torch.Tensor,
    rank: int,
    comm_idx: int,
    stream: Optional[Stream] = None,
    timeout_ms: Optional[int] = None,
):
    """
    Broadcast a tensor from the source rank to all other ranks in the communication group.
    """
    check_tensor(tensor)
    cosmos_c.nccl_broadcast(
        tensor=get_tensor_ptr(tensor),
        count=tensor.numel(),
        dtype=NcclDataType.from_torch(tensor.dtype),
        rank=rank,
        comm_idx=comm_idx,
        stream=get_cuda_stream(stream),
        timeout_ms=get_nccl_timeout_ms() if timeout_ms is None else timeout_ms,
    )


def nccl_send(
    tensor: torch.Tensor,
    peer: int,
    comm_idx: int,
    stream: Optional[Stream] = None,
    timeout_ms: Optional[int] = None,
):
    """
    Send a tensor to a peer in the communication group.
    """
    check_tensor(tensor)
    cosmos_c.nccl_send(
        tensor=get_tensor_ptr(tensor),
        count=tensor.numel(),
        dtype=NcclDataType.from_torch(tensor.dtype),
        peer=peer,
        comm_idx=comm_idx,
        stream=get_cuda_stream(stream),
        timeout_ms=get_nccl_timeout_ms() if timeout_ms is None else timeout_ms,
    )


def nccl_recv(
    tensor: torch.Tensor,
    peer: int,
    comm_idx: int,
    stream: Optional[Stream] = None,
    timeout_ms: Optional[int] = None,
):
    """
    Receive a tensor from a peer in the communication group.
    """
    check_tensor(tensor)
    cosmos_c.nccl_recv(
        tensor=get_tensor_ptr(tensor),
        count=tensor.numel(),
        dtype=NcclDataType.from_torch(tensor.dtype),
        peer=peer,
        comm_idx=comm_idx,
        stream=get_cuda_stream(stream),
        timeout_ms=get_nccl_timeout_ms() if timeout_ms is None else timeout_ms,
    )


def nccl_allreduce(
    sendbuff: torch.Tensor,
    recvbuff: torch.Tensor,
    op: ReduceOp,
    comm_idx: int,
    stream: Optional[Stream] = None,
    timeout_ms: Optional[int] = None,
):
    """
    Allreduce a tensor in the communication group.
    """
    check_tensor(sendbuff)
    check_tensor(recvbuff)
    cosmos_c.nccl_allreduce(
        sendbuff=get_tensor_ptr(sendbuff),
        recvbuff=get_tensor_ptr(recvbuff),
        count=sendbuff.numel(),
        dtype=NcclDataType.from_torch(sendbuff.dtype),
        op=NcclRedOp.to_nccl_reduce_op(op),
        comm_idx=comm_idx,
        stream=get_cuda_stream(stream),
        timeout_ms=get_nccl_timeout_ms() if timeout_ms is None else timeout_ms,
    )


def nccl_alltoall(
    sendbuff: torch.Tensor,
    recvbuff: torch.Tensor,
    comm_idx: int,
    stream: Optional[Stream] = None,
    timeout_ms: Optional[int] = None,
):
    """
    Alltoall a tensor in the communication group.
    """
    check_tensor(sendbuff)
    check_tensor(recvbuff)
    cosmos_c.nccl_alltoall(
        sendbuff=get_tensor_ptr(sendbuff),
        recvbuff=get_tensor_ptr(recvbuff),
        total_size=sendbuff.numel(),
        dtype=NcclDataType.from_torch(sendbuff.dtype),
        comm_idx=comm_idx,
        stream=get_cuda_stream(stream),
        timeout_ms=get_nccl_timeout_ms() if timeout_ms is None else timeout_ms,
    )
