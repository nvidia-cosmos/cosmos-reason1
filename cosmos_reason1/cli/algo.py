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

from cosmos_reason1.utils.network_util import make_request_with_retry
from cosmos_reason1.cli.utils import get_base_url
from cosmos_reason1.utils.api_suffix import COSMOS_API_META_SUFFIX
from cosmos_reason1.cli.custom_group import ControllerNeedeGroup


@click.group(cls=ControllerNeedeGroup)
def algo():
    """
    Query information about training config from controller.
    """
    pass


@algo.command(name="config")
def config(controller_host: str, controller_port: str):
    """
    Show the configuration of the controller.
    """
    base_url = get_base_url(controller_host, controller_port)
    url = urljoin(base_url, COSMOS_API_META_SUFFIX)
    response = make_request_with_retry(
        requests.get,
        [url],
        max_retries=4,
    )
    config = response.json()

    from rich.pretty import pprint

    pprint(config)


def add_command(cli_group):
    cli_group.add_command(algo)
