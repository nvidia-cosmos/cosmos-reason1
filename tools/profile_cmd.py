#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


import requests
import argparse
from cosmos_reason1.utils.network_util import make_request_with_retry
from cosmos_reason1.utils.logging import logger
from functools import partial
import socket


def get_ip_from_hostname(hostname):
    try:
        ip_address = socket.gethostbyname(hostname)
    except socket.error as e:
        logger.error(f"Failed to get IP from hostname: {e}")
        raise e
    return ip_address


def profile_cmd(controller_ip: str, replica_name: str):
    if "localhost" in controller_ip:
        hostname, port = controller_ip.split(":")
        ip_address = get_ip_from_hostname(hostname)
        controller_ip = f"{ip_address}:{port}"
    try:
        controller_ip = [f"http://{controller_ip}/api/set_profile"]
        logger.info(f"Controller IP: {controller_ip}")
        response = make_request_with_retry(
            partial(
                requests.post,
                json={"replica_name": replica_name},
            ),
            controller_ip,
            max_retries=20,
        )
    except Exception as e:
        logger.error(f"Failed to send profile command to controller: {e}")
        raise e
    response = response.json()

    msg = response["message"]
    logger.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A tool to send profile command to a certain policy replica to controller."
    )
    parser.add_argument(
        "-c",
        "--controller_ip",
        type=str,
        required=True,
        help="The address of the controller.",
    )
    parser.add_argument(
        "-r",
        "--replica_name",
        type=str,
        required=True,
        help="The name of the policy replica.",
    )

    args = parser.parse_args()

    profile_cmd(args.controller_ip, args.replica_name)
