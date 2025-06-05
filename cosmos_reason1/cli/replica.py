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
from .utils import console
from urllib.parse import urljoin

from cosmos_reason1.utils.network_util import make_request_with_retry
from cosmos_reason1.cli.utils import (
    create_table_for_all_replicas,
    create_table_for_single_replica,
    get_base_url,
)
from cosmos_reason1.utils.api_suffix import COSMOS_API_STATUS_SUFFIX
from cosmos_reason1.cli.custom_group import ControllerNeedeGroup


@click.group(cls=ControllerNeedeGroup)
def replica():
    """
    Query replica information from controller.
    """
    pass


@replica.command(name="ls")
def list(controller_host: str, controller_port: str):
    """
    List all the replicas that registered to the controller. Including policy and rollout replicas.
    """
    base_url = get_base_url(controller_host, controller_port)
    url = urljoin(base_url, COSMOS_API_STATUS_SUFFIX)
    response = make_request_with_retry(
        requests.get,
        [url],
        max_retries=4,
    )
    replica_dict = response.json()

    replicas_list = []
    if "policy_replicas" in replica_dict:
        policy_replicas = replica_dict["policy_replicas"]
        replicas_list.extend(policy_replicas)

    if "rollout_replicas" in replica_dict:
        rollout_replicas = replica_dict["rollout_replicas"]
        replicas_list.extend(rollout_replicas)

    if len(replicas_list) == 0:
        console.print("No replicas registered to the controller.")
        return
    console.print(create_table_for_all_replicas(replicas_list))


@replica.command(name="lsr")
@click.argument("replica_name")
def list_replica(replica_name: str, controller_host: str, controller_port: str):
    """
    List the details of a specific replica.
    """
    base_url = get_base_url(controller_host, controller_port)
    url = urljoin(base_url, COSMOS_API_STATUS_SUFFIX)
    response = make_request_with_retry(
        requests.get,
        [url],
        max_retries=4,
    )
    replica_dict = response.json()
    current_replica = None

    for replica in replica_dict["policy_replicas"]:
        if replica["name"] == replica_name:
            current_replica = replica
            break
    for replica in replica_dict["rollout_replicas"]:
        if replica["name"] == replica_name:
            current_replica = replica
            break
    if current_replica is None:
        console.print(f"Replica {replica_name} not found.")
        return
    console.print(create_table_for_single_replica(current_replica))


def add_command(cli_group):
    cli_group.add_command(replica)
