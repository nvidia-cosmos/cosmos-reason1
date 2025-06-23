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

from pydantic import BaseModel, model_validator
from typing import List, Dict, Any, Optional


MESH_NAMES: List[str] = ["pp", "dp_shard", "cp", "tp"]


class Role:
    POLICY = "POLICY"
    ROLLOUT = "ROLLOUT"
    REFERENCE = "REFERENCE"
    ALL = [POLICY, ROLLOUT, REFERENCE]


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


class HandshakeInitiatorRequest(BaseModel):
    unique_pair_name: str
    handle_base64: str


class HandshakeAcceptorRequest(BaseModel):
    unique_pair_name: str


class TrainAckRequest(BaseModel):
    replica_name: str
    weight_step: int
    # For profiling
    profile_finished: bool = False
    # For logger report data
    report_data: Dict[str, Any] = {}


class WeightReadyRequest(BaseModel):
    replica_name: str


class ValidationReportRequest(BaseModel):
    src_replica_name: str
    validation_step: int
    prompt_idxs: List[int]
    payloads: List[Any]
    completions: List[List[str]]
    is_end: bool = False
    reference_answer: Optional[str] = None


class RolloutRequest(BaseModel):
    src_replica_name: str
    prompt_idxs: List[int]
    payloads: List[Any]
    completions: List[List[str]]
    is_end: bool = False
    reference_answer: Optional[str] = None


class UnregisterRequest(BaseModel):
    replica_name: str


class HeartbeatRequest(BaseModel):
    replica_name: str


class SetProfileRequest(HeartbeatRequest):
    active_steps: int = 1
    rank_filter: List[int] = [
        0,
    ]
    record_shape: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_modules: bool = False


class SetTracePathRequest(HeartbeatRequest):
    trace_path: str
    global_rank: int


class RegisterRequest(BaseModel):
    replica_name: str
    role: str
    mesh_names: List[str]
    global_rank: int
    host_ip: str
    host_name: str
    ranks: List[int]
    group_size: List[int]

    @model_validator(mode="after")
    def validate_mesh_names(self):
        assert (
            len(self.mesh_names) == len(self.ranks) == len(self.group_size)
        ), "mesh_names, ranks, and group_size must have the same length"
        assert set(self.mesh_names) <= set(
            MESH_NAMES
        ), "mesh_names must be a subset of MESH_NAMES"
        assert (
            self.role in Role.ALL
        ), "role must be one of POLICY, ROLLOUT, or REFERENCE"
        assert (
            self.replica_name is not None and len(self.replica_name) > 0
        ), "replica_name must be a non-empty string being consistent within a replica"

        new_mesh_names = []
        new_ranks = []
        new_group_size = []
        original_map = {}

        # Rebuild the map with deterministic order
        for mesh_name, rank, group_size in zip(
            self.mesh_names, self.ranks, self.group_size
        ):
            original_map[mesh_name] = (rank, group_size)

        for mesh_name in MESH_NAMES:
            new_mesh_names.append(mesh_name)
            new_ranks.append(original_map[mesh_name][0])
            new_group_size.append(original_map[mesh_name][1])

        self.mesh_names = new_mesh_names
        self.ranks = new_ranks
        self.group_size = new_group_size
        return self


class NcclErrRequest(BaseModel):
    replica_name: str
    error: str
