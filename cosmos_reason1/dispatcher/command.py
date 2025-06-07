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

from typing import Dict, List, Optional, Type, Callable
import copy
import uuid
from abc import ABC
from strenum import StrEnum
import msgpack

from cosmos_reason1.dispatcher.replica import Replica
from cosmos_reason1.dispatcher.protocol import Role
from cosmos_reason1.dispatcher.status import RolloutStatusManager, RolloutStatus
from cosmos_reason1.utils.redis_stream import RedisStreamHandler


class CommandType(StrEnum):
    WEIGHT_RESUME = "WEIGHT_RESUME"
    BUILD_MESH = "BUILD_MESH"
    POLICY_TO_POLICY_BROADCAST = "POLICY_TO_POLICY_BROADCAST"
    POLICY_TO_POLICY_UNICAST = "POLICY_TO_POLICY_UNICAST"
    POLICY_TO_ROLLOUT_UNICAST = "POLICY_TO_ROLLOUT_UNICAST"
    ROLLOUT_TO_ROLLOUT_BROADCAST = "ROLLOUT_TO_ROLLOUT_BROADCAST"
    DATA_FETCH = "DATA_FETCH"
    ALL_REDUCE = "ALL_REDUCE"
    STOP = "STOP"
    DUMMY = "DUMMY"


class CommandScope:
    GLOBAL = 0
    LOCAL = 1


class Command(ABC):
    uuid: str

    def __init__(self, uuid: str, scope: CommandScope, command_type: CommandType):
        self.uuid = uuid
        self.scope = scope
        self.command_type = command_type

    def _serialize(self) -> Dict:
        dict_v = copy.deepcopy(self.__dict__)
        return dict_v

    def pack(self):
        return msgpack.packb(self.__dict__)

    @classmethod
    def new_uuid(cls):
        return str(uuid.uuid4())

    @classmethod
    def deserialize(cls, dict_v):
        sub_cls = None
        if dict_v["command_type"] == CommandType.WEIGHT_RESUME:
            sub_cls = WeightResumeCommand
        elif dict_v["command_type"] == CommandType.BUILD_MESH:
            sub_cls = BuildMeshCommand
        elif dict_v["command_type"] == CommandType.POLICY_TO_POLICY_BROADCAST:
            sub_cls = PolicyToPolicyBroadcastCommand
        elif dict_v["command_type"] == CommandType.POLICY_TO_POLICY_UNICAST:
            sub_cls = PolicyToPolicyUnicastCommand
        elif dict_v["command_type"] == CommandType.POLICY_TO_ROLLOUT_UNICAST:
            sub_cls = PolicyToRolloutUnicastCommand
        elif dict_v["command_type"] == CommandType.ROLLOUT_TO_ROLLOUT_BROADCAST:
            sub_cls = RolloutToRolloutBroadcastCommand
        elif dict_v["command_type"] == CommandType.DATA_FETCH:
            sub_cls = DataFetchCommand
        elif dict_v["command_type"] == CommandType.ALL_REDUCE:
            sub_cls = AllReduceCommand
        elif dict_v["command_type"] == CommandType.STOP:
            sub_cls = StopCommand

        if sub_cls is None:
            return DummyCommand
        else:
            return sub_cls.from_dict(dict_v)

    @classmethod
    def depack(cls, byte):
        dict = msgpack.unpackb(byte)
        return cls.deserialize(dict)

    @classmethod
    def from_dict(cls, dict_v: Dict):
        raise NotImplementedError("from_dict is not implemented for Base Command")


class DummyCommand(Command):
    def __int__(self):
        super().__init__("", CommandScope.LOCAL, CommandType.DUMMY)


class StopCommand(Command):
    def __init__(self, replica_name: str, uuid: str):
        super().__init__(uuid, CommandScope.LOCAL, CommandType.STOP)
        self.replica_name = replica_name

    replica_name: Optional[str] = None

    @classmethod
    def trigger(cls, replica: Replica, redis_handler: RedisStreamHandler):
        cmd = cls(replica.name, cls.new_uuid())
        redis_handler.publish_command(cmd.pack(), replica.name)

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(
            dict_v["replica_name"],
            dict_v["uuid"],
        )


class WeightResumeCommand(Command):
    def __init__(self, replica_name: str, uuid: str):
        super().__init__(uuid, CommandScope.LOCAL, CommandType.WEIGHT_RESUME)
        self.replica_name = replica_name

    replica_name: Optional[str] = None

    @classmethod
    def trigger(cls, replica: Replica, redis_handler: RedisStreamHandler):
        assert (
            replica.role == Role.POLICY
        ), "WeightResumeCommand can only be triggered on policy replicas"
        cmd = cls(replica.name, cls.new_uuid())
        redis_handler.publish_command(cmd.pack(), replica.name)
        # initial weight step
        replica.weight_step = 0

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(
            dict_v["replica_name"],
            dict_v["uuid"],
        )


class BuildMeshCommand(Command):
    def __init__(self, replica_name_to_rank: Dict[str, int], uuid: str):
        super().__init__(uuid, CommandScope.GLOBAL, CommandType.BUILD_MESH)
        self.replica_name_to_rank = replica_name_to_rank

    replica_name_to_rank: Dict[str, int]

    @classmethod
    def trigger(cls, replicas: List[Replica], redis_handler: RedisStreamHandler):
        index = 0
        assert all(
            replica.all_atoms_arrived for replica in replicas
        ), "All replicas must have arrived"
        replica_name_to_rank = {}
        for replica in replicas:
            replica_name_to_rank[replica.name] = index
            index += 1
        cmd = cls(replica_name_to_rank, cls.new_uuid())
        for replica in replicas:
            redis_handler.publish_command(cmd.pack(), replica.name)
            replica.in_mesh = True
        return replica_name_to_rank

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(
            dict_v["replica_name_to_rank"],
            dict_v["uuid"],
        )


class PolicyToPolicyBroadcastCommand(Command):
    """
    Only used for policy weight init during initialization. (After `WeightResumeCommand` on `src_replica_name`)
    """

    def __init__(self, src_replica_name: str, dst_replica_names: List[str], uuid: str):
        super().__init__(
            uuid, CommandScope.GLOBAL, CommandType.POLICY_TO_POLICY_BROADCAST
        )
        self.src_replica_name = src_replica_name
        self.dst_replica_names = dst_replica_names

    src_replica_name: Optional[str] = None
    dst_replica_names: List[str]

    @classmethod
    def trigger(
        cls,
        src_replica: Replica,
        dst_replicas: List[Replica],
        redis_handler: RedisStreamHandler,
    ):
        # dst_replicas will contains the src_replica
        cmd = cls(
            src_replica.name, [replica.name for replica in dst_replicas], cls.new_uuid()
        )
        for replica in dst_replicas:
            redis_handler.publish_command(cmd.pack(), replica.name)
            replica.weights_loaded_in_view_of_command = True

            replica.weight_step = src_replica.weight_step

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(
            dict_v["src_replica_name"],
            dict_v["dst_replica_names"],
            dict_v["uuid"],
        )


class PolicyToPolicyUnicastCommand(Command):
    """
    Used for policy dynamic scaling.
    """

    def __init__(self, src_replica_name: str, dst_replica_name: str, uuid: str):
        super().__init__(uuid, CommandScope.LOCAL, CommandType.POLICY_TO_POLICY_UNICAST)
        self.src_replica_name = src_replica_name
        self.dst_replica_name = dst_replica_name

    src_replica_name: Optional[str] = None
    dst_replica_name: Optional[str] = None

    @classmethod
    def trigger(
        cls,
        src_replica: Replica,
        dst_replica: Replica,
        redis_handler: RedisStreamHandler,
    ):
        cmd = cls(src_replica.name, dst_replica.name, cls.new_uuid())
        redis_handler.publish_command(cmd.pack(), src_replica.name)
        redis_handler.publish_command(cmd.pack(), dst_replica.name)
        dst_replica.weights_loaded_in_view_of_command = True

        dst_replica.weight_step = src_replica.weight_step

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(
            dict_v["src_replica_name"],
            dict_v["dst_replica_name"],
            dict_v["uuid"],
        )


class PolicyToRolloutUnicastCommand(Command):
    """
    Used for:
        - weight updating of rollout for on-policy training.
        - weight initialization of first rollout replica.
    """

    do_weight_sync_check_flag: bool = True

    def __init__(
        self,
        src_replica_name: str,
        dst_replica_name: str,
        src_replica_size: int,
        dst_replica_size: int,
        uuid: str,
        do_weight_sync_check: bool = False,
    ):
        super().__init__(
            uuid, CommandScope.LOCAL, CommandType.POLICY_TO_ROLLOUT_UNICAST
        )
        self.src_replica_name = src_replica_name
        self.dst_replica_name = dst_replica_name
        self.src_replica_size = src_replica_size
        self.dst_replica_size = dst_replica_size
        self.do_weight_sync_check = do_weight_sync_check

    src_replica_name: Optional[str] = None
    dst_replica_name: Optional[str] = None
    src_replica_size: Optional[int] = None
    dst_replica_size: Optional[int] = None
    do_weight_sync_check: Optional[bool] = None

    @classmethod
    def trigger(
        cls,
        src_replica: Replica,  # Policy Replica
        dst_replica: Replica,  # Rollout Replica
        src_replica_size: int,
        dst_replica_size: int,
        redis_handler: RedisStreamHandler,
        optimize_step: int,
        status_manager: RolloutStatusManager,
    ):
        cmd = cls(
            src_replica.name,
            dst_replica.name,
            src_replica_size,
            dst_replica_size,
            cls.new_uuid(),
            cls.do_weight_sync_check_flag,
        )
        redis_handler.publish_command(cmd.pack(), src_replica.name)
        redis_handler.publish_command(cmd.pack(), dst_replica.name)

        dst_replica.weight_step = optimize_step
        status_manager.set_optimize_step(dst_replica.name, optimize_step)
        # set the weight ready.
        status_manager.set_status(dst_replica.name, RolloutStatus.READY)

        dst_replica.weights_loaded_in_view_of_command = True

        if cls.do_weight_sync_check_flag:
            cls.do_weight_sync_check_flag = False

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(
            dict_v["src_replica_name"],
            dict_v["dst_replica_name"],
            dict_v["src_replica_size"],
            dict_v["dst_replica_size"],
            dict_v["uuid"],
            dict_v["do_weight_sync_check"],
        )


class RolloutToRolloutBroadcastCommand(Command):
    """
    Used for rollout weight update.(After `PolicyToRolloutUnicastCommand` on `src_replica_name`)
    """

    def __init__(self, src_replica_name: str, dst_replica_names: List[str], uuid: str):
        super().__init__(
            uuid, CommandScope.GLOBAL, CommandType.ROLLOUT_TO_ROLLOUT_BROADCAST
        )
        self.src_replica_name = src_replica_name
        self.dst_replica_names = dst_replica_names

    src_replica_name: Optional[str] = None
    dst_replica_names: List[str]

    @classmethod
    def trigger(
        cls,
        src_replica: Replica,
        dst_replicas: List[Replica],
        redis_handler: RedisStreamHandler,
        optimize_step: int,
        status_manager: RolloutStatusManager,
    ):
        # dst_replicas will contains the src_replica
        if not src_replica.in_mesh:
            return
        cmd = cls(
            src_replica.name, [replica.name for replica in dst_replicas], cls.new_uuid()
        )
        for replica in dst_replicas:
            if not replica.in_mesh:
                continue
            redis_handler.publish_command(cmd.pack(), replica.name)
            replica.weights_loaded_in_view_of_command = True
            replica.weight_step = optimize_step
            status_manager.set_optimize_step(replica.name, optimize_step)
            status_manager.set_status(replica.name, RolloutStatus.READY)

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(
            dict_v["src_replica_name"],
            dict_v["dst_replica_names"],
            dict_v["uuid"],
        )


class DataFetchCommand(Command):
    """
    Used to fetch data from the controller.
    items_count: int,  Number of items to fetch.
    replica_name: str, Name of the replica to fetch data.
    """

    def __init__(
        self,
        replica_name: str,
        items_count: int,
        global_step: int,
        total_steps: int,
        remain_samples_num: int,
        uuid: str,
        # For profiling
        do_profile: bool,
        active_steps: int,
        rank_filter: List[int],
        record_shape: bool,
        profile_memory: bool,
        with_stack: bool,
        with_modules: bool,
    ):
        super().__init__(uuid, CommandScope.LOCAL, CommandType.DATA_FETCH)
        self.replica_name = replica_name
        self.items_count = items_count
        self.global_step = global_step
        self.total_steps = total_steps
        self.remain_samples_num = remain_samples_num

        # Profling config
        self.do_profile = do_profile
        self.active_steps = active_steps
        self.rank_filter = rank_filter
        self.record_shape = record_shape
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules

    replica_name: Optional[str] = None
    items_count: Optional[int] = None
    global_step: Optional[int] = None
    total_steps: Optional[int] = None
    remain_samples_num: Optional[int] = None

    do_profile: Optional[bool] = None
    active_steps: Optional[int] = None
    rank_filter: Optional[List[int]] = None
    record_shape: Optional[bool] = None
    profile_memory: Optional[bool] = None
    with_stack: Optional[bool] = None
    with_modules: Optional[bool] = None

    @classmethod
    def trigger(
        cls,
        replica: Replica,
        items_count: int,
        global_step: int,
        total_steps: int,
        remain_samples_num: int,
        redis_handler: RedisStreamHandler,
    ):
        cmd = cls(
            replica.name,
            items_count,
            global_step,
            total_steps,
            remain_samples_num,
            cls.new_uuid(),
            replica.sub_profiler_config.do_profile,
            replica.sub_profiler_config.active_steps,
            replica.sub_profiler_config.rank_filter,
            replica.sub_profiler_config.record_shape,
            replica.sub_profiler_config.profile_memory,
            replica.sub_profiler_config.with_stack,
            replica.sub_profiler_config.with_modules,
        )
        redis_handler.publish_command(cmd.pack(), replica.name)

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(
            dict_v["replica_name"],
            dict_v["items_count"],
            dict_v["global_step"],
            dict_v["total_steps"],
            dict_v["remain_samples_num"],
            dict_v["uuid"],
            # For profiling
            dict_v["do_profile"],
            dict_v["active_steps"],
            dict_v["rank_filter"],
            dict_v["record_shape"],
            dict_v["profile_memory"],
            dict_v["with_stack"],
            dict_v["with_modules"],
        )


class AllReduceCommand(Command):
    """
    Used to perform all-reduce operation.
    replica_name_to_rank: Dict[str, int], Mapping of replica names to ranks.
    """

    def __init__(self, replica_name_to_rank: Dict[str, int], uuid: str):
        super().__init__(uuid, CommandScope.LOCAL, CommandType.ALL_REDUCE)
        self.replica_name_to_rank = replica_name_to_rank

    replica_name_to_rank: Dict[str, int]

    @classmethod
    def trigger(
        cls, replica_name_to_rank: Dict[str, int], redis_handler: RedisStreamHandler
    ):
        cmd = cls(replica_name_to_rank, cls.new_uuid())
        for replica_name, _ in replica_name_to_rank.items():
            redis_handler.publish_command(cmd.pack(), replica_name)

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(
            dict_v["replica_name_to_rank"],
            dict_v["uuid"],
        )


class CommandRegistry:
    def __init__(self):
        self.registry: Dict[Type[Command], Callable] = {}

    def register(self, key: Type[Command], func: Callable):
        if key not in self.registry:
            self.registry[key] = func
        else:
            raise ValueError(f"Command handler for {key} already registered")

    def get_command_handler(self, key: Type[Command]) -> Optional[Callable]:
        return self.registry.get(key, None)
