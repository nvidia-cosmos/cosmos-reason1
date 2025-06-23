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
    VALIDATE = "VALIDATE"
    DUMMY = "DUMMY"


class CommandScope:
    GLOBAL = 0
    LOCAL = 1


class Command(ABC):
    uuid_value: str

    def __init__(
        self,
        scope: CommandScope,
        command_type: CommandType,
        uuid_value: Optional[str] = None,
    ):
        self.uuid_value = uuid_value if uuid_value is not None else str(uuid.uuid4())
        self.scope = scope
        self.command_type = command_type

    def _serialize(self) -> Dict:
        dict_v = copy.deepcopy(self.__dict__)
        return dict_v

    def pack(self):
        return msgpack.packb(self.__dict__)

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

        if sub_cls is None:
            raise ValueError(f"Unknown command type: {dict_v['command_type']}")
        else:
            return sub_cls.from_dict(dict_v)

    @classmethod
    def depack(cls, byte):
        dict = msgpack.unpackb(byte)
        return cls.deserialize(dict)

    @classmethod
    def from_dict(cls, dict_v: Dict):
        raise NotImplementedError("from_dict is not implemented for Base Command")


class WeightResumeCommand(Command):
    def __init__(self, replica_name: str, **kwargs):
        kwargs["scope"] = CommandScope.LOCAL
        kwargs["command_type"] = CommandType.WEIGHT_RESUME
        super().__init__(**kwargs)
        self.replica_name = replica_name

    replica_name: str

    @classmethod
    def trigger(cls, replica: Replica, redis_handler: RedisStreamHandler):
        assert (
            replica.role == Role.POLICY
        ), "WeightResumeCommand can only be triggered on policy replicas"
        cmd = cls(replica.name)
        redis_handler.publish_command(cmd.pack(), replica.name)
        # initial weight step
        replica.weights_loaded_in_view_of_command = True

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(**dict_v)


class BuildMeshCommand(Command):
    def __init__(self, replica_name_to_rank: Dict[str, int], **kwargs):
        kwargs["scope"] = CommandScope.GLOBAL
        kwargs["command_type"] = CommandType.BUILD_MESH
        super().__init__(**kwargs)
        self.replica_name_to_rank = replica_name_to_rank

    replica_name_to_rank: Dict[str, int]

    @classmethod
    def trigger(cls, replicas: List[Replica], redis_handler: RedisStreamHandler):
        index = 0
        assert all(
            replica.all_atoms_arrived for replica in replicas
        ), "All replicas must have arrived"
        replicas_to_rank = {}
        for replica in replicas:
            replica.status.mesh_rank = index
            replicas_to_rank[replica.name] = index
            index += 1
        cmd = cls(replicas_to_rank)
        for replica in replicas:
            redis_handler.publish_command(cmd.pack(), replica.name)

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(**dict_v)


class PolicyToPolicyBroadcastCommand(Command):
    """
    Only used for policy weight init during initialization. (After `WeightResumeCommand` on `src_replica_name`)
    """

    def __init__(self, src_replica_name: str, dst_replica_names: List[str], **kwargs):
        kwargs["scope"] = CommandScope.GLOBAL
        kwargs["command_type"] = CommandType.POLICY_TO_POLICY_BROADCAST
        super().__init__(**kwargs)
        self.src_replica_name = src_replica_name
        self.dst_replica_names = dst_replica_names

    src_replica_name: str
    dst_replica_names: List[str]

    @classmethod
    def trigger(
        cls,
        src_replica: Replica,
        dst_replicas: List[Replica],
        redis_handler: RedisStreamHandler,
    ):
        # dst_replicas will contains the src_replica
        cmd = cls(src_replica.name, [replica.name for replica in dst_replicas])
        for replica in dst_replicas:
            redis_handler.publish_command(cmd.pack(), replica.name)
            replica.weights_loaded_in_view_of_command = True

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(**dict_v)


class PolicyToPolicyUnicastCommand(Command):
    """
    Used for policy dynamic scaling.
    """

    def __init__(self, src_replica_name: str, dst_replica_name: str, **kwargs):
        kwargs["scope"] = CommandScope.LOCAL
        kwargs["command_type"] = CommandType.POLICY_TO_POLICY_UNICAST
        super().__init__(**kwargs)
        self.src_replica_name = src_replica_name
        self.dst_replica_name = dst_replica_name

    src_replica_name: str
    dst_replica_name: str

    @classmethod
    def trigger(
        cls,
        src_replica: Replica,
        dst_replica: Replica,
        redis_handler: RedisStreamHandler,
    ):
        cmd = cls(src_replica.name, dst_replica.name)
        redis_handler.publish_command(cmd.pack(), src_replica.name)
        redis_handler.publish_command(cmd.pack(), dst_replica.name)
        dst_replica.weights_loaded_in_view_of_command = True

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(**dict_v)


class PolicyToRolloutUnicastCommand(Command):
    """
    Used for:
        - weight updating of rollout for on-policy training.
        - weight initialization of first rollout replica.
    """

    _do_weight_sync_check_flag: bool = True

    def __init__(
        self,
        src_replica_name: str,
        dst_replica_name: str,
        src_replica_size: int,
        dst_replica_size: int,
        do_weight_sync_check: bool = False,
        weight_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        **kwargs,
    ):
        kwargs["scope"] = CommandScope.LOCAL
        kwargs["command_type"] = CommandType.POLICY_TO_ROLLOUT_UNICAST
        super().__init__(**kwargs)
        self.src_replica_name = src_replica_name
        self.dst_replica_name = dst_replica_name
        self.src_replica_size = src_replica_size
        self.dst_replica_size = dst_replica_size
        self.do_weight_sync_check = do_weight_sync_check
        self.weight_step = weight_step
        self.total_steps = total_steps

    src_replica_name: str
    dst_replica_name: str
    src_replica_size: int
    dst_replica_size: int
    do_weight_sync_check: bool
    weight_step: Optional[int]
    total_steps: Optional[int]

    @classmethod
    def trigger(
        cls,
        src_replica: Replica,  # Policy Replica
        dst_replica: Replica,  # Rollout Replica
        src_replica_size: int,
        dst_replica_size: int,
        weight_step: Optional[int],
        total_steps: Optional[int],
        redis_handler: RedisStreamHandler,
    ):
        cmd = cls(
            src_replica.name,
            dst_replica.name,
            src_replica_size,
            dst_replica_size,
            cls._do_weight_sync_check_flag,
            weight_step,
            total_steps,
        )
        redis_handler.publish_command(cmd.pack(), src_replica.name)
        redis_handler.publish_command(cmd.pack(), dst_replica.name)
        dst_replica.weights_loaded_in_view_of_command = True

        if cls._do_weight_sync_check_flag:
            cls._do_weight_sync_check_flag = False

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(**dict_v)


class RolloutToRolloutBroadcastCommand(Command):
    """
    Used for rollout weight update.(After `PolicyToRolloutUnicastCommand` on `src_replica_name`)
    """

    def __init__(
        self,
        src_replica_name: str,
        dst_replica_names: List[str],
        weight_step: Optional[int],
        total_steps: Optional[int],
        **kwargs,
    ):
        kwargs["scope"] = CommandScope.GLOBAL
        kwargs["command_type"] = CommandType.ROLLOUT_TO_ROLLOUT_BROADCAST
        super().__init__(**kwargs)
        self.src_replica_name = src_replica_name
        self.dst_replica_names = dst_replica_names
        self.weight_step = weight_step
        self.total_steps = total_steps

    src_replica_name: str
    dst_replica_names: List[str]
    weight_step: Optional[int]
    total_steps: Optional[int]

    @classmethod
    def trigger(
        cls,
        src_replica: Replica,
        dst_replicas: List[Replica],
        weight_step: Optional[int],
        total_steps: Optional[int],
        redis_handler: RedisStreamHandler,
    ):
        # dst_replicas will contains the src_replica
        if not src_replica.in_mesh:
            return
        cmd = cls(
            src_replica.name,
            [replica.name for replica in dst_replicas],
            weight_step,
            total_steps,
        )
        for replica in dst_replicas:
            if not replica.in_mesh:
                continue
            redis_handler.publish_command(cmd.pack(), replica.name)
            replica.weights_loaded_in_view_of_command = True

    def replica_should_stop(self):
        if self.weight_step is not None and self.total_steps is not None:
            return self.weight_step >= self.total_steps
        return False

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(**dict_v)


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
        # For profiling
        do_profile: Optional[bool] = None,
        active_steps: Optional[int] = None,
        rank_filter: Optional[List[int]] = None,
        record_shape: Optional[bool] = None,
        profile_memory: Optional[bool] = None,
        with_stack: Optional[bool] = None,
        with_modules: Optional[bool] = None,
        **kwargs,
    ):
        kwargs["scope"] = CommandScope.LOCAL
        kwargs["command_type"] = CommandType.DATA_FETCH
        super().__init__(**kwargs)
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

    replica_name: str
    items_count: int
    global_step: Optional[int]
    total_steps: Optional[int]
    remain_samples_num: int

    do_profile: bool
    active_steps: int
    rank_filter: List[int]
    record_shape: bool
    profile_memory: bool
    with_stack: bool
    with_modules: bool

    @classmethod
    def trigger(
        cls,
        replica: Replica,
        items_count: int,
        global_step: Optional[int],
        total_steps: Optional[int],
        remain_samples_num: int,
        redis_handler: RedisStreamHandler,
    ):
        cmd = cls(
            replica.name,
            items_count,
            global_step,
            total_steps,
            remain_samples_num,
            replica.sub_profiler_config.do_profile,
            replica.sub_profiler_config.active_steps,
            replica.sub_profiler_config.rank_filter,
            replica.sub_profiler_config.record_shape,
            replica.sub_profiler_config.profile_memory,
            replica.sub_profiler_config.with_stack,
            replica.sub_profiler_config.with_modules,
        )
        redis_handler.publish_command(cmd.pack(), replica.name)

    def replica_should_stop(self):
        if self.global_step is not None and self.total_steps is not None:
            return self.global_step >= self.total_steps
        return False

    @classmethod
    def from_dict(cls, dict_v: Dict):
        return cls(**dict_v)


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
