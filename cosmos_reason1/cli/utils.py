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

"""
Common utilities for the CLI.
"""

import requests
import sys
from typing import Dict, List
from rich.console import Console
from rich.table import Table

from cosmos_reason1.utils.network_util import make_request_with_retry
from cosmos_reason1.utils.api_suffix import COSMOS_API_STATUS_SUFFIX
from urllib.parse import urljoin

console = Console()


def check_controller_running(base_url: str):
    """
    Check if the controller is running on the given address and port.
    Args:
        address (str): The address of the controller.
        port (str): The port of the controller.
    Returns:
        bool: True if the controller is running, False otherwise.
    """
    url = urljoin(base_url, COSMOS_API_STATUS_SUFFIX)
    is_running = False
    try:
        response = make_request_with_retry(
            requests.get,
            [url],
            max_retries=4,
        )
        if response.status_code == 200:
            is_running = True
        else:
            is_running = False
    except Exception as e:
        console.print(f"Exception when checking controller running: {e}")
        is_running = False
    if not is_running:
        console.print(f"Please check if the controller is running on {base_url}")
        sys.exit(1)
    return is_running


def get_base_url(controller_host: str, controller_port: str):
    base_url = f"http://{controller_host}:{controller_port}"
    if not check_controller_running(base_url):
        console.print(f"Controller is not running on {base_url}.")
        sys.exit(1)
    return base_url


def create_table_for_single_replica(replica: Dict):
    table = Table(
        title=f"Replica : {replica['name']}, Role: {replica['role']}",
        show_lines=True,
    )

    table.add_column("Atom Global Rank")
    table.add_column("Host IP")
    table.add_column("Host Name")
    table.add_column("Trace Path")

    for atom in replica["atoms"]:
        atom_global_rank = atom["global_rank"]
        host_ip = atom["host_ip"]
        host_name = atom["host_name"]
        trace_path = atom["trace_path"]

        table.add_row(
            str(atom_global_rank),
            host_ip,
            host_name,
            trace_path if trace_path else "None",
        )

    return table


def create_table_for_all_replicas(replicas: List[Dict]):
    """
    Create simplified table for all replicas.
    """
    table = Table(
        title="Replicas",
        show_lines=True,
    )
    # One Row for each replica
    table.add_column("Replica")
    table.add_column("Role")
    table.add_column("Arrived")
    table.add_column("Atom Number")
    table.add_column("Weight Step")
    table.add_column("Pending Rollouts")

    for replica in replicas:
        if replica["arrived"]:
            arrived_string = "[green]Yes[/green]"
        else:
            arrived_string = "[red]No[/red]"
        atom_number = len(replica["atoms"])
        if replica["role"] == "POLICY":
            pending_rollouts = replica["pending_rollouts"]
        else:
            pending_rollouts = "N/A"
        table.add_row(
            replica["name"],
            replica["role"],
            arrived_string,
            str(atom_number),
            str(replica["weight_step"]),
            str(pending_rollouts),
        )
    return table
