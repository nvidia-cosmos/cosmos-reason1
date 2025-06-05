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

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import math
from cosmos_reason1.dispatcher.protocol import Role, MESH_NAMES, RegisterRequest
from cosmos_reason1.policy.config import SubProfilerConfig
import asyncio
from cosmos_reason1.dispatcher.algo.base import RuleBasedAlgo
import weakref
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.redis_stream import RedisStreamHandler
import msgpack


@dataclass
class Rollout:
    payload: Any
    completion: str
    extra_info: Dict[str, Any]
    reward: float
    advantage: float
    prompt_idx: int

    def __init__(
        self,
        payload: Any,
        completion: str,
        extra_info: Dict[str, Any],
        reward: float,
        advantage: float,
        prompt_idx: int,
    ):
        self.payload = payload
        self.completion = completion
        self.extra_info = extra_info
        self.reward = reward
        self.advantage = advantage
        self.prompt_idx = prompt_idx


class RolloutGroup:
    """
    RolloutGroup is a data structure that contains the prompt and completions of a rollout.
    For MutliModal-LM, image/video/audio could be included in the extra_info.
    """

    def __init__(
        self,
        prompt_idx: int,
        payload: Any,
        completions: List[str],
        extra_info: Dict[str, Any],
        reference_answer: str,
    ):
        self.prompt_idx: int = prompt_idx
        self.payload: Any = payload
        self.completions: List[str] = completions
        self.extra_info: Dict[str, Any] = extra_info
        self.reference_answer: str = reference_answer

    def compute_rollouts(self, algo: RuleBasedAlgo) -> List[Rollout]:
        assert (
            self.reference_answer is not None
        ), "[RolloutGroup] Reference answer is not provided"

        rewards = [
            algo.compute_reward(completion, self.reference_answer)
            for completion in self.completions
        ]
        logger.debug(f"[RolloutGroup] Rewards: {rewards}")
        advantages = algo.compute_advantage(rewards)
        logger.debug(f"[RolloutGroup] Advantages: {advantages}")

        return [
            Rollout(
                payload=self.payload,
                completion=completion,
                extra_info=self.extra_info,
                reward=reward,
                advantage=advantage,
                prompt_idx=self.prompt_idx,
            )
            for completion, reward, advantage in zip(
                self.completions, rewards, advantages
            )
        ]


class BatchedRolloutGroup:
    """
    Batched Wrapper of the RolloutGroup
    """

    def __init__(self):
        self.rollout_groups: List[RolloutGroup] = []

    def __len__(self):
        return len(self.rollout_groups)

    def __getitem__(self, idx: int) -> RolloutGroup:
        return self.rollout_groups[idx]

    def __setitem__(self, idx: int, rollout_group: RolloutGroup):
        self.rollout_groups[idx] = rollout_group

    def __delitem__(self, idx: int):
        del self.rollout_groups[idx]

    @classmethod
    def from_rollout_groups(
        cls, rollout_groups: List[RolloutGroup]
    ) -> "BatchedRolloutGroup":
        batched_rollout_group = cls()
        batched_rollout_group.rollout_groups = rollout_groups
        return batched_rollout_group


@dataclass
class Atom:
    """
    Atom is the smallest unit of a computation mesh.
    Usually it is a single GPU process.

    replica_name: str Name of the replica that this atom belongs to
    ranks: List[int] Rank of each dimension in the order of MESH_NAMES
    group_size: List[int] Size of each dimension in the order of MESH_NAMES
    global_rank: int global rank of the atom in this replica.
    host_ip: str IP address of the host machine that this atom belongs to.
    host_name: str Name of the host machine that this atom belongs to.
    trace_path: str Path to the trace file of this atom/rank that stores. May be s3 or local path.
    """

    replica_name: str
    ranks: List[int]
    global_rank: int
    host_ip: str
    host_name: str
    trace_path: Optional[str]
    group_size: List[int]
    rollout_: asyncio.Queue[Rollout]

    def __init__(
        self,
        global_rank: int,
        host_ip: str,
        host_name: str,
        trace_path: str,
        ranks: List[int],
        group_size: List[int],
        replica_name: str,
    ):
        self.ranks = ranks
        self.group_size = group_size
        self.replica_name = replica_name
        self.rollout_queue: asyncio.Queue[Rollout] = asyncio.Queue()
        self._replica: Optional[weakref.ReferenceType["Replica"]] = None
        self.global_rank = global_rank
        self.host_ip = host_ip
        self.host_name = host_name
        self.trace_path = trace_path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ranks": self.ranks,
            "group_size": self.group_size,
            "replica_name": self.replica_name,
            "global_rank": self.global_rank,
            "host_ip": self.host_ip,
            "host_name": self.host_name,
            "trace_path": self.trace_path,
            "replica": self._replica().name if self._replica else "None",
        }

    @property
    def replica(self) -> "Replica":
        assert self._replica is not None, f"Atom {self} is not bound to a replica"
        return self._replica()

    def bind_replica(self, replica: "Replica"):
        self._replica = weakref.ref(replica)

    async def put_rollout(self, rollout: Rollout):
        assert (
            self.tp_rank() == 0 and self.cp_rank() == 0 and self.pp_rank() == 0
        ), f"Atom {self} is not a dp_shard master atom, cannot put rollout"
        self.rollout_queue.put_nowait(rollout)

    async def fetch_rollouts(self) -> List[Rollout]:
        assert (
            self.tp_rank() == 0 and self.cp_rank() == 0 and self.pp_rank() == 0
        ), f"Atom {self} is not a dp_shard master atom, cannot fetch rollouts"
        rollouts = []
        while not self.rollout_queue.empty():
            rollouts.append(self.rollout_queue.get_nowait())
        return rollouts

    @classmethod
    def from_register_request(cls, request: RegisterRequest) -> "Atom":
        return cls(
            global_rank=request.global_rank,
            host_ip=request.host_ip,
            host_name=request.host_name,
            trace_path=None,
            ranks=request.ranks,
            group_size=request.group_size,
            replica_name=request.replica_name,
        )

    def tp_rank(self) -> int:
        return self.ranks[MESH_NAMES.index("tp")]

    def cp_rank(self) -> int:
        return self.ranks[MESH_NAMES.index("cp")]

    def pp_rank(self) -> int:
        return self.ranks[MESH_NAMES.index("pp")]

    def dp_shard_rank(self) -> int:
        return self.ranks[MESH_NAMES.index("dp_shard")]

    def __post_init__(self):
        assert (
            len(self.ranks) == len(self.group_size)
        ), f"Ranks and group_size must have the same length, got {len(self.ranks)} and {len(self.group_size)}"
        assert (
            len(self.ranks) == len(MESH_NAMES)
        ), f"Ranks must have the same length as MESH_NAMES, got {len(self.ranks)} and {len(MESH_NAMES)}"

    def __str__(self):
        return f"{self.replica_name}_{self.global_rank}"


@dataclass
class Replica:
    """
    Replica is a single `DP Relicate` unit, where sub:
        - TP
        - CP
        - PP
        - DP_SHARD
    could be used to form a `DP Relicate` unit.

    name: str Name of the replica
    role: Role Role of the replica, either POLICY or ROLLOUT
    atoms: Dict[str, Atom] Sub-units of the replica
    """

    name: str
    role: Role
    atoms: Dict[str, Atom]
    command_queue: asyncio.Queue
    weights_loaded_in_view_of_command: bool = False
    in_mesh: bool = False
    pending_rollouts: int = 0
    weight_step: int = -1

    # For profiling
    sub_profiler_config: SubProfilerConfig = field(default_factory=SubProfilerConfig)

    def __init__(self, name: str, role: Role, atoms: List[Atom]):
        self.name = name  # Note: name must be unique across all the replicas.
        self.role = role
        self.atoms = {str(atom): atom for atom in atoms}
        self.command_queue = asyncio.Queue()
        self.pending_rollouts = 0
        self.weight_step = -1

        self.sub_profiler_config = SubProfilerConfig()

    def to_dict(self) -> Dict[str, Any]:
        atoms = []
        for atom_key, atom in self.atoms.items():
            atoms.append(atom.to_dict())
        return {
            "name": self.name,
            "role": self.role,
            "atoms": atoms,
            "arrived": self.all_atoms_arrived,
            "weight_step": self.weight_step,
            "pending_rollouts": self.pending_rollouts,
        }

    def get_atom(self, ranks: List[int]) -> Atom:
        assert (
            len(ranks) == len(MESH_NAMES)
        ), f"Ranks must have the same length as MESH_NAMES, got {len(ranks)} and {len(MESH_NAMES)}"
        return self.atoms[str(ranks)]

    def arrive(self, atom: Atom):
        assert str(atom) not in self.atoms, f"Atom {atom} already exists"
        # Verify group size is consistent with existing atoms
        if len(self.atoms) > 0:
            existing_atom = next(iter(self.atoms.values()))
            assert (
                atom.group_size == existing_atom.group_size
            ), f"Atom {atom} has inconsistent group size"
        self.atoms[str(atom)] = atom
        if self.role == Role.POLICY and self.all_atoms_arrived:
            # Check out all the dp_shard atoms
            # Only those atoms are responsible rollout fetching
            #    1. pipeline rank is 0
            #    2. dp_shard rank is any
            #    3. cp rank is 0
            #    4. tp rank is 0
            any_atom = next(iter(self.atoms.values()))
            group_sizes = any_atom.group_size
            assert (
                len(group_sizes) == len(MESH_NAMES)
            ), f"Group sizes must have the same length as MESH_NAMES, got {len(group_sizes)} and {len(MESH_NAMES)}"
            tp_size, cp_size, dp_shard_size, pp_size = 1, 1, 1, 1

            for i in range(len(MESH_NAMES)):
                if MESH_NAMES[i] == "tp":
                    tp_size = group_sizes[i]
                elif MESH_NAMES[i] == "cp":
                    cp_size = group_sizes[i]
                elif MESH_NAMES[i] == "pp":
                    pp_size = group_sizes[i]
                elif MESH_NAMES[i] == "dp_shard":
                    dp_shard_size = group_sizes[i]
                else:
                    raise ValueError(f"Unknown mesh name: {MESH_NAMES[i]}")
            assert (
                tp_size * cp_size * dp_shard_size * pp_size == len(self.atoms)
            ), f"Group sizes must be consistent with the number of atoms, got {tp_size} * {cp_size} * {dp_shard_size} * {pp_size} = {tp_size * cp_size * dp_shard_size * pp_size} and {len(self.atoms)}"

            dp_shard_atoms = []

            for atom in self.atoms.values():
                if atom.tp_rank() == 0 and atom.cp_rank() == 0 and atom.pp_rank() == 0:
                    dp_shard_atoms.append(atom)
            assert (
                len(dp_shard_atoms) == dp_shard_size
            ), f"Number of dp_shard atoms must be equal to dp_shard size, got {len(dp_shard_atoms)} and {dp_shard_size}"
            self.dp_shard_atoms = dp_shard_atoms
            self.round_robin_atom = dp_shard_atoms[-1]

    @property
    def all_atoms_arrived(self) -> bool:
        assert len(self.atoms) > 0, f"Replica {self.name} has no atoms"
        atom = next(iter(self.atoms.values()))
        # Dot product of ranks and group_size should be equal
        return len(self.atoms) == math.prod(atom.group_size)

    async def put_rollout(self, rollout: Rollout, redis_handler: RedisStreamHandler):
        assert (
            self.role == Role.POLICY
        ), f"Replica {self.name} is not a policy replica, cannot put rollout"
        assert (
            self.all_atoms_arrived
        ), f"Replica {self.name} tries to put rollout but not all atoms have arrived"
        # Check which atom should handle this rollout
        # Publish the rollout to the redis stream to be consumed by policy replicas
        redis_handler.publish_rollout(msgpack.packb(rollout.__dict__), self.name)
        self.pending_rollouts += 1

    async def find_atom(self, global_rank: int) -> Atom:
        key = f"{self.name}_{global_rank}"
        return self.atoms.get(key)

    async def set_trace_path(self, trace_path: str, global_rank: int):
        atom = await self.find_atom(global_rank)
        logger.info(f"[Controller]: Set trace path of atom {atom} to {trace_path}")
        atom.trace_path = trace_path
        return str(atom)

    def __eq__(self, other):
        return isinstance(other, Replica) and self.name == other.name

    def __hash__(self):
        return hash(self.name)
