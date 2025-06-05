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
import requests
from urllib.parse import urljoin
from rich.table import Table

from cosmos_reason1.utils.api_suffix import COSMOS_API_NCCL_COMM_GET_ALL_SUFFIX
from cosmos_reason1.utils.network_util import make_request_with_retry
from cosmos_reason1.cli.utils import console, get_base_url
from cosmos_reason1.utils.util import b64_to_list
from cosmos_reason1.cli.custom_group import ControllerNeedeGroup


@click.group(cls=ControllerNeedeGroup)
def nccl():
    """
    Query information about NCCL from controller.
    """
    pass


@nccl.command(name="ls")
def comm_all(controller_host: str, controller_port: str):
    """
    List all the NCCL communicators stored in the controller.
    """
    base_url = get_base_url(controller_host, controller_port)
    url = urljoin(base_url, COSMOS_API_NCCL_COMM_GET_ALL_SUFFIX)
    response = make_request_with_retry(
        requests.get,
        [url],
        max_retries=4,
    )

    comm_info = response.json()["comm_info"]
    comm_dict = {}
    for key, value in comm_info.items():
        comm_dict[key] = str(b64_to_list(value))

    table = Table(title="NCCL Communicator", show_lines=True)
    table.add_column("Src")
    table.add_column("Dst")
    table.add_column("Paired Rank")
    table.add_column("Value")
    for key, value in comm_dict.items():
        replicas = key.split("_")
        if len(replicas) == 3:
            src_replica, dst_replica, paired_rank = replicas
        else:
            src_replica, dst_replica = replicas[:2]
            paired_rank = "N/A"
        table.add_row(src_replica, dst_replica, paired_rank, value)
    console.print(table)


def add_command(cli_group):
    cli_group.add_command(nccl)
