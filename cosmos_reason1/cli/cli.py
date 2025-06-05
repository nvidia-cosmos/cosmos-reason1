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

import click
import socket
import sys

from . import replica
from . import profiler
from . import nccl
from . import algo
from .utils import console


@click.group()
def cosmos():
    """
    Cosmos Reason1 CLI.
    """
    pass


replica.add_command(cosmos)
profiler.add_command(cosmos)
nccl.add_command(cosmos)
algo.add_command(cosmos)


def get_ip_from_hostname(hostname):
    try:
        ip_address = socket.gethostbyname(hostname)
    except socket.error as e:
        console.print(f"Failed to get IP from hostname: {e}")
        sys.exit(1)
    return ip_address


# @cosmos.command()
# @click.option(
#     "--controller-host", "-ch", help="The host that runs the controller", default=None
# )
# @click.option(
#     "--controller-port",
#     "-cp",
#     help="The port that points to the controller",
#     default=None,
# )
# def register(controller_host: str, controller_port: str):
#     """
#     Register the controller url to the local cache.
#     """
#     if "localhost" in controller_host:
#         controller_host = get_ip_from_hostname(controller_host)

#     base_url = f"http://{controller_host}:{controller_port}"
#     if not check_controller_running(base_url):
#         console.print(f"Controller is not running on {base_url}")
#         sys.exit(1)

#     controller_port = int(controller_port)

#     LocalCacheForController.set_and_save(controller_host, controller_port)

#     console.print(
#         f"Controller url successfully registered to file: {LocalCacheForController.CONTROLLER_URL_FILE}"
#     )


if __name__ == "__main__":
    cosmos()
