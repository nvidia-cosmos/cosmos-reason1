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

# Standard library imports
import array
import fcntl
import math
import os
import re
import socket
import time
import struct
from datetime import timedelta
from typing import Dict, Iterable, Optional, Union
from functools import partial
from contextlib import contextmanager
from importlib.metadata import version

# Third party imports
import requests
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, distribute_module, Placement
from torch.distributed.tensor.parallel import ParallelStyle


# Local imports
import cosmos_reason1._cpp as cosmos_c
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.util import make_request_with_retry
from cosmos_reason1.utils import constant


def init_distributed(cpu_enabled: bool):
    world_size = os.environ.get("WORLD_SIZE", 1)
    if world_size == 1:
        return
    torch.distributed.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        timeout=timedelta(seconds=600),
    )


def get_controller_metadata() -> Dict:
    """
    Get metadata from the controller with retry logic.

    Returns:
        Tuple containing (remote_ips, remote_port, metadata)
    """
    remote_hosts = os.environ["COSMOS_CONTROLLER_HOST"]
    # Verify in the format of host:port
    remote_ips, remote_port = remote_hosts.split(":")
    remote_ips = remote_ips.split(";")
    for remote_ip in remote_ips:
        if not re.match(
            r"^([a-zA-Z0-9_.-]+):([1-9][0-9]{0,4})$", f"{remote_ip}:{remote_port}"
        ):
            raise ValueError(f"Invalid remote host: {remote_ip}:{remote_port}")
    remote_hosts = [
        f"http://{remote_ip}:{remote_port}/api/meta" for remote_ip in remote_ips
    ]
    try:
        r = make_request_with_retry(
            requests.get,
            remote_hosts,
            max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
        )
    except Exception as e:
        logger.error(f"Failed to communicate with controller after attempts: {e}")
        raise e
    metadata: Dict = r.json()
    remote_eth_ips = metadata.get("config", {}).get("eth_ips", [])
    if remote_eth_ips:
        remote_ips = remote_ips + remote_eth_ips.split(";")
    return remote_ips, remote_port, metadata


def destroy_distributed():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


NCCL_REDUCE_OPS = {
    "sum": 0,
    "prod": 1,
    "max": 2,
    "min": 3,
    "avg": 4,
}


@torch.no_grad()
def gradient_reduce_across_dp_replicas_(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    comm_idx: int,
) -> torch.Tensor:
    """
    Reduce a tensor across data parallel replicas.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will reduce gradients.
        comm_idx (int): The nccl communicator id for the reduction.
    """
    if comm_idx < 0:
        # comm not initialized, skip reduction
        return

    grads = [p.grad for p in parameters if p.grad is not None]

    # We only need to reduce DTensor's local grad, this is to avoid tensor.grad == nullptr
    for i, g in enumerate(grads):
        if isinstance(g, DTensor):
            grads[i] = g.to_local()

    # create bucket for all grads, we can allreduce them in one go
    # NOTE: why we don't set DTensor as bucket view?
    # This is becuase we can't be sure that the training framework
    # never release grad, or clean grad by set None.
    # Create temporary bucket is a more reliable solution.
    buckets: dict[torch.dtype, list[torch.Tensor]] = {}
    for g in grads:
        if g.dtype not in buckets:
            buckets[g.dtype] = []
        buckets[g.dtype].append(g.flatten())

    # move all grad into one bucket
    nranks = cosmos_c.get_nccl_comm_count(comm_idx)

    op = NCCL_REDUCE_OPS.get("sum")
    for bucket in buckets.values():
        tmp_buffer = torch.cat(bucket, dim=0).contiguous()
        # we need scale grad by 1/nranks to keep grad is mean at sample level
        tmp_buffer = tmp_buffer / nranks
        cosmos_c.nccl_allreduce(tmp_buffer, tmp_buffer, op, comm_idx)

        # copy the result back to original grad
        offset = 0
        for g in bucket:
            size = g.numel()
            g.copy_(tmp_buffer[offset : offset + size].view_as(g))
            offset += size
            assert offset <= tmp_buffer.numel(), "offset should be equal to total size"


@torch.no_grad()
def gradient_norm_clipping(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    pp_mesh: Optional[DeviceMesh] = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = (
        torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
        if len(grads) > 0
        else torch.tensor(0.0).to(torch.cuda.current_device()).float()
    )

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.

        # Remove FT replicate dimension if it exists.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def _dist_reduce(x: torch.Tensor, reduceOp: str, mesh: DeviceMesh) -> float:
    if isinstance(x, DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()
    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


def dist_max(x: torch.Tensor, mesh: DeviceMesh) -> float:
    return _dist_reduce(x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh)


def dist_mean(x: torch.Tensor, mesh: DeviceMesh) -> float:
    return _dist_reduce(x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh)


def get_ip_address(ifname):
    """
    Returns the IPv4 address assigned to the given interface.

    Args:
        ifname (str): The interface name (e.g., "eth0").

    Returns:
        str or None: The IPv4 address as a string if found, else None.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 0x8915 is SIOCGIFADDR; pack interface name (limited to 15 chars)
        ip_bytes = fcntl.ioctl(
            s.fileno(), 0x8915, struct.pack("256s", ifname[:15].encode("utf-8"))
        )
        ip = socket.inet_ntoa(ip_bytes[20:24])
        return ip
    except OSError:
        return None


def get_mellanox_ips():
    """
    Scans for Mellanox Ethernet interfaces (vendor "0x15b3", "0x1d0f") in /sys/class/net and returns
    their associated IPv4 addresses.

    Returns:
        list of dict: Each dict contains keys 'eth' (interface name) and 'ip' (IPv4 address).
    """
    result = []
    net_dir = "/sys/class/net"

    if not os.path.isdir(net_dir):
        return result

    for iface in os.listdir(net_dir):
        vendor_path = os.path.join(net_dir, iface, "device", "vendor")
        if not os.path.isfile(vendor_path):
            continue
        try:
            with open(vendor_path, "r") as vf:
                vendor = vf.read().strip()
        except Exception:
            continue

        # Amazon: 0x1d0f
        # Mellanox: 0x15b3
        if vendor not in ["0x1d0f", "0x15b3"]:
            continue

        # Get the IPv4 address for this interface.
        ip = get_ip_address(iface)
        if ip is not None:
            result.append({"eth": iface, "ip": ip})
    return result


def get_all_ipv4_addresses():
    """
    Returns all IPv4 addresses for interfaces on the system, excluding 127.0.0.1.

    Uses the SIOCGIFCONF ioctl call to fetch all interfaces.

    Returns:
        list of dict: Each dict contains 'eth' (interface name) and 'ip' (IPv4 address).
    """
    ip_list = []
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Allocate buffer for maximum number of interfaces.
    max_interfaces = 128
    bytes_size = max_interfaces * 32
    names = array.array("B", b"\0" * bytes_size)

    # SIOCGIFCONF to get list of interfaces.
    try:
        outbytes = struct.unpack(
            "iL",
            fcntl.ioctl(
                s.fileno(),
                0x8912,  # SIOCGIFCONF
                struct.pack("iL", bytes_size, names.buffer_info()[0]),
            ),
        )[0]
    except Exception:
        logger.error("Failed to get all IPv4 addresses")
        return ip_list

    namestr = names.tobytes()

    # Each entry is typically 40 bytes.
    for i in range(0, outbytes, 40):
        iface_name = namestr[i : i + 16].split(b"\0", 1)[0].decode("utf-8")
        ip_addr = socket.inet_ntoa(namestr[i + 20 : i + 24])
        if ip_addr != "127.0.0.1":
            ip_list.append({"eth": iface_name, "ip": ip_addr})
    return ip_list


def get_eth_ips():
    """
    Determines whether the Infiniband driver is active.

    - If /sys/class/infiniband exists, returns the IP addresses bound to Mellanox Ethernet interfaces.
    - Otherwise, returns all IPv4 addresses on the system except 127.0.0.1.

    Returns:
        list of dict: Each dictionary contains 'eth' (interface name) and 'ip' (IPv4 address).
    """
    infiniband_dir = "/sys/class/infiniband"

    ip_info = []

    if os.path.isdir(infiniband_dir):
        # Infiniband is active; return Mellanox interface IPs.
        ip_info = get_mellanox_ips()

    if not ip_info:
        # Infiniband not found; return all IPv4 addresses (excluding loopback).
        ip_info = get_all_ipv4_addresses()

    return [x["ip"] for x in ip_info]


class ReplicateParallel(ParallelStyle):
    def __init__(
        self, *, use_local_output: bool = True, input_layout: Optional[Placement] = None
    ):
        super().__init__()
        self.use_local_output = use_local_output
        self.input_layout = input_layout or Replicate()

    def _replicate_module_fn(
        self, name: str, module: torch.nn.Module, device_mesh: DeviceMesh
    ):
        for p_name, param in module.named_parameters():
            replicated_param = torch.nn.Parameter(
                DTensor.from_local(param, device_mesh, [Replicate()], run_check=False)
            )
            module.register_parameter(p_name, replicated_param)

    @staticmethod
    def _prepare_input_fn(input_layout, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if isinstance(input_tensor, DTensor):
            return input_tensor
        elif isinstance(input_tensor, torch.Tensor):
            # assume the input passed in already sharded on the sequence dim and create the DTensor
            return DTensor.from_local(
                input_tensor, device_mesh, [input_layout], run_check=False
            )
        else:
            raise ValueError(
                f"expecting input of {mod} to be a torch.Tensor or DTensor, but got {input_tensor}"
            )

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        if isinstance(outputs, tuple):
            return tuple([o.to_local() if use_local_output else o for o in outputs])
        else:
            return outputs.to_local() if use_local_output else outputs

    def _apply(
        self, module: torch.nn.Module, device_mesh: DeviceMesh
    ) -> torch.nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
            partial(self._prepare_input_fn, self.input_layout),
            partial(self._prepare_output_fn, self.use_local_output),
        )


def broadcast_object_cpu(obj, src=0, device=torch.device("cpu"), group=None):
    """
    Broadcast an object from the source process to all processes.
    The object is first converted to a list and then broadcasted.
    """
    self_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size == 1:
        return obj

    obj_lst = [obj if self_rank == src else None]
    dist.broadcast_object_list(obj_lst, src=src, device=device, group=group)
    return obj_lst[0]


@contextmanager
def nccl_timeout_watchdog(wait_stream=False):
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
        logger.warning(
            "NCCL version is less than 2.26.2, which is known to hang in some cases, please upgrade to a newer version"
        )

    start_time = time.time()
    threshold_ms = cosmos_c.nccl_timeout_in_ms()
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
