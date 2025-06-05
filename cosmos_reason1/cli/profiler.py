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
import sys


from functools import partial
from urllib.parse import urljoin

from cosmos_reason1.utils.network_util import make_request_with_retry
from cosmos_reason1.cli.utils import console, get_base_url
from cosmos_reason1.utils.api_suffix import COSMOS_API_SET_PROFILE_SUFFIX
from cosmos_reason1.cli.custom_group import ControllerNeedeGroup


@click.group(cls=ControllerNeedeGroup)
def profile():
    """
    Manage profiler behavior of the training.
    """
    pass


def comma_separated_int_list(ctx, param, value):
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",")]


@profile.command(name="set")
@click.argument("replica_name", type=str)
@click.option(
    "--active-steps", "-as", type=int, help="The number of steps that profiler traces."
)
@click.option(
    "--rank-filter",
    "-rf",
    callback=comma_separated_int_list,
    help="The ranks that profiler traces, separated by comma.",
)
@click.option(
    "--record-shape",
    "-rs",
    is_flag=True,
    help="Whether to record the shape of the tensor in the profiler.",
)
@click.option(
    "--profile-memory",
    "-pm",
    is_flag=True,
    help="Whether to profile the memory of the tensor in the profiler.",
)
@click.option(
    "--with-stack",
    "-ws",
    is_flag=True,
    help="Whether to record the stack trace of the profiler.",
)
@click.option(
    "--with-modules",
    "-wm",
    is_flag=True,
    help="Whether to record the modules of the profiler.",
)
def set_profile(
    replica_name,
    active_steps,
    rank_filter,
    record_shape,
    profile_memory,
    with_stack,
    with_modules,
    controller_host: str,
    controller_port: str,
):
    """
    Set profile mode of the replica named REPLICA_NAME.
    """
    base_url = get_base_url(controller_host, controller_port)
    url = urljoin(base_url, COSMOS_API_SET_PROFILE_SUFFIX)
    if len(rank_filter) == 0:
        rank_filter = [
            0,
        ]
    if active_steps is None:
        active_steps = 1
    response = make_request_with_retry(
        partial(
            requests.post,
            json={
                "replica_name": replica_name,
                "active_steps": active_steps,
                "rank_filter": rank_filter,
                "record_shape": record_shape,
                "profile_memory": profile_memory,
                "with_stack": with_stack,
                "with_modules": with_modules,
            },
        ),
        [url],
        max_retries=4,
    )

    if response.status_code != 200:
        console.print(f"Failed to set profile for replica: {replica_name}")
        sys.exit(1)

    msg = response.json()["message"]
    console.print(msg)


def add_command(cli_group):
    cli_group.add_command(profile)
