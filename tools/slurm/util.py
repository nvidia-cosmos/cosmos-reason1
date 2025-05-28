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

import json
import copy
from typing import List, Literal
from dataclasses import dataclass, asdict

@dataclass
class ReplicaLaunchMetadata:
    # The number of nodes for the specific replica
    nnode: int
    # The role of the specific replica
    role: Literal["policy", "rollout"]
    # The head node of specific replica
    rendezvous_node: int
    # Port for rendezvous, in case there are multiple replicas on the same node to avoid port conflicts
    rendezvous_port: int
    # The number of GPUs visible to the specific replica
    visible_gpus: List[int]

    def __init__(self, nnode: int, role: Literal["policy", "rollout"], rendezvous_node: int, rendezvous_port: int, visible_gpus: List[int]):
        self.nnode = nnode
        self.role = role
        self.rendezvous_node = rendezvous_node
        self.rendezvous_port = rendezvous_port
        self.visible_gpus = visible_gpus

@dataclass
class NodeLaunchMetadata:
    colocation: List[ReplicaLaunchMetadata]

    def __init__(self, colocation: List[ReplicaLaunchMetadata]):
        self.colocation = colocation

    def to_json(self):
        return asdict(self)

    @staticmethod
    def from_json_list(json_str: str) -> List["NodeLaunchMetadata"]:
        dicts = json.loads(json_str)
        return [NodeLaunchMetadata(
            colocation=[ReplicaLaunchMetadata(**x) for x in d["colocation"]]
        ) for d in dicts]
