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

import time
import random
import socket
import fcntl
import struct
import array
import os
from typing import Any, Callable, List, Union

from cosmos_reason1.utils.constant import COSMOS_HTTP_RETRY_CONFIG
from cosmos_reason1.utils.logging import logger


def get_local_ip():
    """
    Get the local IP address of the machine.

    Returns:
        Local IP address as a string
    """
    try:
        import socket

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return [local_ip, hostname]
    except Exception as e:
        logger.error(f"Error getting local IP address: {e}")
        return None


def status_check_for_response(response):
    """
    Handle the status code for the response.
    Raises an exception if the status code is not 200.
    """
    response.raise_for_status()


def make_request_with_retry(
    requests: Union[Callable, List[Callable]],
    urls: List[str] = None,
    response_parser: Callable = status_check_for_response,
    max_retries: int = COSMOS_HTTP_RETRY_CONFIG.max_retries,
    retries_per_delay: int = COSMOS_HTTP_RETRY_CONFIG.retries_per_delay,
    initial_delay: float = COSMOS_HTTP_RETRY_CONFIG.initial_delay,
    max_delay: float = COSMOS_HTTP_RETRY_CONFIG.max_delay,
    backoff_factor: float = COSMOS_HTTP_RETRY_CONFIG.backoff_factor,
) -> Any:
    """
    Make an HTTP GET request with exponential backoff retry logic.

    Args:
        requests (List[Callable]): The functions to make the request in an alternative way
        urls (List[str]): List of host URLs to try
        response_parser (Callable): Function to parse the response
        max_retries (int): Maximum number of retry attempts
        retries_per_delay (int): Number of retries to attempt at each delay level
        initial_delay (float): Initial delay between retries in seconds
        max_delay (float): Maximum delay between retries in seconds
        backoff_factor (float): Factor to increase delay between retries

    Returns:
        Any: The response object from the successful request or redis client request.

    Raises:
        Exception: If all retry attempts fail
    """
    delay = initial_delay
    last_exception = None
    total_attempts = 0
    url_index = 0
    request_idx = 0

    if isinstance(requests, Callable):
        requests = [requests]

    while total_attempts < max_retries:
        # Try multiple times at the current delay level
        total_retries_cur_delay = 0
        while total_retries_cur_delay < retries_per_delay:
            try:
                request = requests[request_idx]
                if urls is not None:
                    url = urls[url_index]
                    r = request(url)
                else:
                    url = None
                    r = request()
                if response_parser is not None:
                    response_parser(r)
                return r

            except Exception as e:
                last_exception = e
                url_index += 1
                if url_index >= (1 if urls is None else len(urls)):
                    url_index = 0
                    request_idx += 1
                    if request_idx >= len(requests):
                        request_idx = 0
                        total_retries_cur_delay += 1
                        total_attempts += 1
                logger.debug(
                    f"Request failed: {e}. Attempt {total_attempts} of {max_retries} for {request} on {url}."
                )
                if total_attempts >= max_retries:
                    break

                if request_idx != 0 or url_index != 0:
                    jitter = (1.0 + random.random()) * initial_delay
                    time.sleep(jitter)
                    continue
                # Add some jitter to prevent thundering herd
                jitter = (1.0 + random.random()) * delay
                time.sleep(jitter)

        # Increase delay for next round of retries
        delay = min(delay * backoff_factor, max_delay)
    if last_exception is not None:
        raise last_exception
    else:
        raise Exception(f"All retry attempts failed for all urls: {urls}")


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


def find_available_port(start_port):
    max_port = 65535  # Maximum port number
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue

    raise RuntimeError("No available port found in the specified range.")
