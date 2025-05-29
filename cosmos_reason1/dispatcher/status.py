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

from strenum import StrEnum
from typing import Dict, List
from cosmos_reason1.utils.constant import COSMOS_HEARTBEAT_TIMEOUT
from cosmos_reason1.utils.logging import logger


class PolicyStatus(StrEnum):
    """
    Enum for policy status.
    There are 7 statuses:
    UNINITIALIZED: The policy is uninitialized.
    READY: The policy is ready to run.
    RUNNING: The policy is running.
    BACKWARDED: The policy has finished backward.
    REDUCED: The policy has finished reduce.
    END: The policy has finished.
    DELETED: The policy has been deleted.
    """

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    RUNNING = "running"
    BACKWARDED = "backwarded"
    REDUCED = "reduced"
    END = "end"
    DELETED = "deleted"


class LifeStatus(StrEnum):
    """
    Enum for life status.
    """

    ALIVE = "alive"
    DEAD = "dead"


class PolicyStatusManager:
    """
    A class to manage the status of a policy.
    """

    status: Dict[str, PolicyStatus]
    train_batch_per_replica: int
    num_data_samples: int
    total_steps: int
    train_step: Dict[str, int]
    optimize_step: Dict[str, int]
    policy_to_rank: Dict[str, int]
    heartbeat_timestamp: Dict[str, int]
    life_status: Dict[str, LifeStatus]

    def __init__(self):
        self.status = {}
        self.train_batch_per_replica = 0
        self.num_data_samples = 0
        self.total_steps = 0
        self.train_step = {}
        self.policy_to_rank = {}
        self.optimize_step = {}
        self.heartbeat_timestamp = {}
        self.life_status = {}
        self.deleted = {}

    def set_train_batch_per_replica(self, train_batch_per_replica: int):
        """
        Set the train batch per replica for policys.
        """
        self.train_batch_per_replica = train_batch_per_replica

    def set_num_data_samples(self, num_data_samples: int):
        """
        Set the number of data samples for policys.
        """
        self.num_data_samples = num_data_samples

    def set_timestamp(self, replica_name: str, timestamp: int):
        """
        Set the timestamp of the policy.
        """
        self.heartbeat_timestamp[replica_name] = timestamp
        # set the life status to alive
        self.set_life_status(replica_name, LifeStatus.ALIVE)

    def get_timestamp(self, replica_name: str) -> int:
        """
        Get the timestamp of the policy.
        """
        if replica_name not in self.heartbeat_timestamp:
            raise ValueError(f"[Controller]Policy {replica_name} not found")
        return self.heartbeat_timestamp[replica_name]

    def set_life_status(self, replica_name: str, life_status: LifeStatus):
        """
        Set the life status of the policy.
        """
        self.life_status[replica_name] = life_status

    def get_life_status(self, replica_name: str) -> LifeStatus:
        """
        Get the life status of the policy.
        """
        if replica_name not in self.life_status:
            raise ValueError(f"[Controller]Policy {replica_name} not found")
        return self.life_status[replica_name]

    def maintain_life_status(self, now):
        """
        Maintain the life status of the rollout.
        """
        dead_replicas = set()
        for replica_name in self.life_status:
            if now - self.get_timestamp(replica_name) > COSMOS_HEARTBEAT_TIMEOUT:
                logger.warning(f"[Controller] Policy {replica_name} is dead")
                self.set_life_status(replica_name, LifeStatus.DEAD)
                dead_replicas.add(replica_name)

        return dead_replicas

    def set_status(self, name: str, status: PolicyStatus):
        """
        Set the status of the policy.
        """
        if name not in self.status:
            assert (
                status == PolicyStatus.UNINITIALIZED
            ), "Policy status should be UNINITIALIZED when first created"
            self.status[name] = status
            self.train_step[name] = self.completed_train_step()
            self.optimize_step[name] = self.completed_optimize_step()
            return
        assert (
            status != PolicyStatus.UNINITIALIZED
        ), "Policy status should not be UNINITIALIZED when already created"
        if status == PolicyStatus.DELETED:
            # Remove the policy from the status
            assert name in self.status, "Policy status not found"
            assert name in self.train_step, "Train step not found"
            assert name in self.optimize_step, "Optimize step not found"
            stats = {
                "status": self.status.pop(name),
                "train_step": self.train_step.pop(name),
                "optimize_step": self.optimize_step.pop(name),
                "heartbeat_timestamp": self.heartbeat_timestamp.pop(name, None),
                "life_status": self.life_status.pop(name, None),
            }
            self.deleted[name] = stats
            return
        self.status[name] = status
        if status == PolicyStatus.BACKWARDED:
            assert name in self.train_step, "Train step not found"
            # Increment the train step
            self.train_step[name] += 1
            self.num_data_samples -= self.train_batch_per_replica
        elif status == PolicyStatus.REDUCED:
            assert name in self.optimize_step, "Optimize step not found"
            # Increment the optimize step
            self.optimize_step[name] += 1

    def set_ranks(self, policy_to_rank: Dict[str, int]):
        """
        Set the ranks of the policies.
        """
        self.policy_to_rank = policy_to_rank
        # Update total step when policy replicas are set
        num_policy_replicas = len(policy_to_rank)
        if num_policy_replicas > 0:
            self.total_steps = self.completed_train_step() + self.num_data_samples // (
                self.train_batch_per_replica * num_policy_replicas
            )

    def get_total_steps(self) -> int:
        """
        Get the total step of the policies.
        """
        if self.total_steps == 0:
            raise ValueError("Total step is not set")
        return self.total_steps

    def remove_from_ranks(self, name: str):
        """
        Remove the policy from the ranks.
        """
        if name in self.policy_to_rank:
            self.policy_to_rank.pop(name)

    def get_world_size(self) -> int:
        """
        Get the world size of the policies.
        """
        return len(self.policy_to_rank)

    def get_status(self, name: str) -> PolicyStatus:
        """
        Get the status of the policy.
        """
        if name not in self.status:
            raise KeyError(f"Policy {name} not found")
        return self.status[name]

    def all_with_status(self, status: List[PolicyStatus]) -> bool:
        """
        Check if all policies have the given status.
        """
        return all([x in status for x in self.status.values()])

    def all_backwarded(self) -> bool:
        """
        Check if all policies are backwarded.
        """
        return self.all_with_status([PolicyStatus.BACKWARDED])

    def all_reduced(self) -> bool:
        """
        Check if all policies are reduced.
        """
        return self.all_with_status([PolicyStatus.REDUCED])

    def all_ready(self) -> bool:
        """
        Check if all policies are ready.
        """
        return self.all_with_status([PolicyStatus.READY])

    def all_ready_or_reduced(self) -> bool:
        """
        Check if all policies are ready or reduced.
        """
        return self.all_with_status([PolicyStatus.READY, PolicyStatus.REDUCED])

    def completed_train_step(self) -> int:
        """
        Get the train step as the minimum of all policies.
        """
        if len(self.train_step) == 0:
            return 0
        return min(self.train_step.values())

    def completed_optimize_step(self) -> int:
        """
        Get the optimize step as the minimum of all policies.
        """
        if len(self.optimize_step) == 0:
            return 0
        return min(self.optimize_step.values())


class RolloutStatus(StrEnum):
    """
    Enum for rollout status.
    There are 4 statuses:
    READY: The rollout is ready to run.
    END: The rollout has finished.
    UNINITIALIZED: The rollout is uninitialized.
    PAUSE: The rollout is paused.
    """

    READY = "ready"
    END = "end"
    UNINITIALIZED = "uninitialized"
    PAUSE = "pause"


class RolloutStatusManager:
    """
    A class to manage the status of rollout replicas.
    """

    optimize_step: Dict[str, int]  # Weight version tracking of rollout.
    heartbeat_timestamp: Dict[str, int]
    life_status: Dict[str, LifeStatus]
    status: Dict[str, RolloutStatus]
    rollout_to_rank: Dict[str, int]

    def __init__(self):
        self.optimize_step = {}
        self.heartbeat_timestamp = {}
        self.life_status = {}
        self.status = {}
        self.rollout_to_rank = {}

    def set_timestamp(self, replica_name: str, timestamp: int):
        """
        Set the timestamp of the rollout.
        """
        self.heartbeat_timestamp[replica_name] = timestamp
        self.set_life_status(replica_name, LifeStatus.ALIVE)

    def get_timestamp(self, replica_name: str) -> int:
        """
        Get the timestamp of the rollout.
        """
        if replica_name not in self.heartbeat_timestamp:
            raise ValueError(f"[Controller] Rollout {replica_name} not found")
        return self.heartbeat_timestamp[replica_name]

    def set_optimize_step(self, replica_name: str, optimize_step: int):
        """
        Set the train step of the rollout.
        """
        logger.debug(
            f"[Controller] Set optimize step of rollout {replica_name} to {optimize_step}"
        )
        self.optimize_step[replica_name] = optimize_step

    def get_optimize_step(self, replica_name: str) -> int:
        """
        Get the train step of the rollout.
        """
        if replica_name not in self.optimize_step:
            raise ValueError(f"[Controller] Rollout {replica_name} not found")
        return self.optimize_step[replica_name]

    def set_life_status(self, replica_name: str, life_status: LifeStatus):
        """
        Set the life status of the rollout.
        """
        self.life_status[replica_name] = life_status

    def get_life_status(self, replica_name: str) -> LifeStatus:
        """
        Get the life status of the rollout.
        """
        if replica_name not in self.life_status:
            raise ValueError(f"[Controller] Rollout {replica_name} not found")
        return self.life_status[replica_name]

    def set_status(self, replica_name: str, status: RolloutStatus):
        """
        Set the status of the rollout.
        """
        if replica_name not in self.status:
            assert (
                status == RolloutStatus.UNINITIALIZED
            ), "rollout status should be UNINITIALIZED when first created"
            self.status[replica_name] = status
            return
        assert (
            status != RolloutStatus.UNINITIALIZED
        ), "rollout status should not be UNINITIALIZED when already created"
        if self.status[replica_name] != RolloutStatus.END:
            self.status[replica_name] = status

    def get_status(self, replica_name: str):
        if replica_name not in self.status:
            raise KeyError(f"Rollout {replica_name} not found")
        return self.status[replica_name]

    def pop(self, replica_name: str):
        """
        Pop the rollout from the status manager.
        """
        if replica_name in self.optimize_step:
            self.optimize_step.pop(replica_name)
        if replica_name in self.heartbeat_timestamp:
            self.heartbeat_timestamp.pop(replica_name)
        if replica_name in self.life_status:
            self.life_status.pop(replica_name)
        if replica_name in self.status:
            self.status.pop(replica_name)

    def maintain_life_status(self, now):
        """
        Maintain the life status of the rollout.
        """
        dead_replicas = set()
        for replica_name in self.life_status:
            if now - self.get_timestamp(replica_name) > COSMOS_HEARTBEAT_TIMEOUT:
                logger.warning(f"[Controller] Rollout {replica_name} is dead")
                self.set_life_status(replica_name, LifeStatus.DEAD)
                dead_replicas.add(replica_name)

        # pop the dead replicas from status manager
        for replica_name in dead_replicas:
            self.pop(replica_name)

        return dead_replicas

    def all_ready(self) -> bool:
        """
        Check if all rollouts are ready.
        """
        return all([x == RolloutStatus.READY for x in self.status.values()])

    def all_end(self) -> bool:
        """
        Check if all rollouts are end.
        """
        return all([x == RolloutStatus.END for x in self.status.values()])

    def set_ranks(self, rollout_to_rank: Dict[str, int]):
        """
        Set the ranks of the rollouts
        """
        self.rollout_to_rank = rollout_to_rank

    def remove_from_ranks(self, name: str):
        """
        Remove the rollout from the ranks.
        """
        if name in self.rollout_to_rank:
            self.rollout_to_rank.pop(name)

    def get_world_size(self) -> int:
        """
        Get the world size of the rollouts.
        """
        return len(self.rollout_to_rank)
