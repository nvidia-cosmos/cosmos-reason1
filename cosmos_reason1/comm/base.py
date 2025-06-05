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

import torch.distributed as dist
import uuid
from typing import Dict, Callable, Type, Optional
import copy
import requests
import time
import atexit
from functools import partial
import threading
from urllib.parse import urljoin
from cosmos_reason1.utils.redis_stream import RedisStreamHandler
from cosmos_reason1.utils.network_util import make_request_with_retry, get_local_ip
from cosmos_reason1.dispatcher.command import CommandRegistry, Command
from cosmos_reason1.utils.logging import logger
import cosmos_reason1.utils.constant as constant
import cosmos_reason1.utils.distributed as dist_utils
from cosmos_reason1.dispatcher.protocol import MESH_NAMES
from cosmos_reason1.policy.model import get_data_packer
from cosmos_reason1.utils.api_suffix import (
    COSMOS_API_REGISTER_SUFFIX,
    COSMOS_API_UNREGISTER_SUFFIX,
)
import base64
import cloudpickle


class CommMixin:
    policy_command_handler_registry = CommandRegistry()
    rollout_command_handler_registry = CommandRegistry()

    @classmethod
    def register_policy_command_handler(cls, command_type: Type[Command]):
        def decorator(func):
            cls.policy_command_handler_registry.register(command_type, func)
            return func

        return decorator

    @classmethod
    def register_rollout_command_handler(cls, command_type: Type[Command]):
        def decorator(func):
            cls.rollout_command_handler_registry.register(command_type, func)
            return func

        return decorator

    @classmethod
    def get_policy_command_handler(
        cls, command_type: Type[Command]
    ) -> Optional[Callable]:
        return cls.policy_command_handler_registry.get_command_handler(command_type)

    @classmethod
    def get_rollout_command_handler(
        cls, command_type: Type[Command]
    ) -> Optional[Callable]:
        return cls.rollout_command_handler_registry.get_command_handler(command_type)

    def init_comm(self):
        self.replica_name = str(dist_utils.broadcast_object_cpu(uuid.uuid4()))
        logger.info(
            f"{self.role} Replica started at global rank {self.global_rank}, with replica name: {self.replica_name}"
        )
        # Fetch metadata from the controller
        # Get list of remote IPs and port
        self.remote_ips, self.remote_port, metadata = (
            dist_utils.get_controller_metadata()
        )

        # `sft_user_dataset` is only used in SFT mode when the user provides a dataset
        self.sft_user_dataset = None
        sft_user_dataset = metadata.get("sft_user_dataset", None)
        if sft_user_dataset:
            sft_user_dataset = base64.b64decode(sft_user_dataset)
            sft_user_dataset = cloudpickle.loads(sft_user_dataset)
            if hasattr(sft_user_dataset, "setup"):
                sft_user_dataset.setup(self.config, self.tokenizer)
            self.sft_user_dataset = sft_user_dataset

        user_data_packer = metadata.get("user_data_packer", None)
        if user_data_packer:
            user_data_packer = base64.b64decode(user_data_packer)
            user_data_packer = cloudpickle.loads(user_data_packer)
            self.data_packer = user_data_packer
            logger.info(f"Using user-provided data packer: {self.data_packer}")
        else:
            self.data_packer = get_data_packer(self.config)
            logger.info(f"Using default data packer: {self.data_packer}")
        self.data_packer.setup(self.config, self.tokenizer)

        self.remote_hosts = [
            f"http://{remote_ip}:{self.remote_port}" for remote_ip in self.remote_ips
        ]
        self.register_to_controller()

    def register_to_controller(self):
        if hasattr(self, "_is_registered"):
            return

        target_mesh_names = copy.deepcopy(MESH_NAMES)
        ranks = []
        group_size = []
        for mesh_name in MESH_NAMES:
            if (
                self.parallel_dims.mesh.mesh_dim_names
                and mesh_name in self.parallel_dims.mesh.mesh_dim_names
            ):
                ranks.append(self.parallel_dims.mesh[mesh_name].get_local_rank())
                group_size.append(self.parallel_dims.mesh[mesh_name].size())
            else:
                ranks.append(0)
                group_size.append(1)

        try:
            host_info_tuple = get_local_ip()
            if host_info_tuple is None:
                raise Exception("Failed to get local IP address")
            host_ip, host_name = host_info_tuple

            payload = {
                "replica_name": self.replica_name,
                "role": self.role,
                "mesh_names": target_mesh_names,
                "ranks": ranks,
                "group_size": group_size,
                "global_rank": self.global_rank,
                "host_ip": host_ip,
                "host_name": host_name,
            }
            make_request_with_retry(
                partial(
                    requests.post,
                    json=payload,
                ),
                self.get_alternative_urls(COSMOS_API_REGISTER_SUFFIX),
                max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to register to controller: {e}")
            raise e

        dist.barrier()  # wait all the atoms registered.

        if self.global_rank == 0:
            logger.info(
                f"{self.role} Replica {self.replica_name} registered to controller"
            )
        self._is_registered = True
        atexit.register(self.unregister_from_controller)

    def unregister_from_controller(self):
        if not hasattr(self, "_is_registered"):
            return
        self._is_registered = False
        # let only rank == 0 send the unregister request
        if self.global_rank == 0:
            try:
                make_request_with_retry(
                    partial(
                        requests.post,
                        json={"replica_name": self.replica_name},
                    ),
                    self.get_alternative_urls(COSMOS_API_UNREGISTER_SUFFIX),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
            except Exception as e:
                logger.error(f"Failed to unregister to controller: {e}")

    def get_group_unique_key(self, replica_name_to_rank: Dict[str, int]):
        return (
            "_".join(
                [
                    k
                    for k, _ in sorted(
                        replica_name_to_rank.items(), key=lambda item: item[1]
                    )
                ]
            )
            + "_"
            + str(self.global_rank)
        )

    def init_redis(self):
        # For command fetch via redis connection
        self.redis_controller = RedisStreamHandler(
            ips=self.remote_ips, port=int(self.config.redis)
        )
        logger.debug(
            f"[{self.role}] Init redis at {self.remote_ips}:{self.redis_controller.port}"
        )

    def heartbeat_trigger(self, shutdown_signal: threading.Event):
        while True:
            if shutdown_signal.is_set():
                logger.info(
                    "[Policy] Heartbeat thread is stopped since the shutdown signal is set."
                )
                break

            try:
                make_request_with_retry(
                    partial(
                        requests.post,
                        json={
                            "replica_name": self.replica_name,
                        },
                    ),
                    self.get_alternative_urls("api/heartbeat"),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
            except Exception as e:
                logger.error(f"Failed to send heartbeat to controller: {e}")
            time.sleep(constant.COSMOS_HEARTBEAT_SEND_INTERVAL)

    def start_heartbeat(self, shutdown_signal: threading.Event):
        if self.global_rank == 0:
            # Start the thread with daemon=True, so it will exit when the main program exits.
            thread = threading.Thread(
                target=self.heartbeat_trigger,
                args=(shutdown_signal,),
                daemon=True,
            )
            thread.start()
            return thread
        else:
            return None

    def get_alternative_urls(self, suffix: str):
        # Get the alternative URLs for the given suffix
        urls = []
        for remote_host in self.remote_hosts:
            urls.append(urljoin(remote_host, suffix))
        return urls
