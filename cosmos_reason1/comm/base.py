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

import torch
import torch.distributed as dist
import uuid
from cosmos_reason1.utils.logging import logger
import cosmos_reason1.utils.constant as constant
import cosmos_reason1.utils.distributed as dist_utils
import requests
from typing import Dict
from cosmos_reason1.dispatcher.protocol import MESH_NAMES
import copy
import time
import atexit
from cosmos_reason1.utils.redis_stream import RedisStreamHandler
import threading


class CommMixin:
    def init_comm(self):
        replica_name = [uuid.uuid4() if self.global_rank == 0 else None]
        dist.broadcast_object_list(replica_name, src=0, device=torch.device("cpu"))
        self.replica_name = str(replica_name[0])
        logger.info(
            f"{self.role} Replica started at global rank {self.global_rank}, with replica name: {self.replica_name}"
        )
        # Fetch metadata from the controller
        self.remote_ip, self.remote_port, metadata = (
            dist_utils.get_controller_metadata()
        )
        self.remote_host = f"http://{self.remote_ip}:{self.remote_port}"
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

        while True:
            counter = 0
            try:
                payload = {
                    "replica_name": self.replica_name,
                    "role": self.role,
                    "mesh_names": target_mesh_names,
                    "ranks": ranks,
                    "group_size": group_size,
                }
                requests.post(f"{self.remote_host}/api/register", json=payload)
            except Exception as e:
                logger.error(f"Failed to register to controller: {e}")
                counter += 1
                if counter > 10:
                    raise e
                time.sleep(1)
                continue
            break

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
            requests.post(
                f"{self.remote_host}/api/unregister",
                json={"replica_name": self.replica_name},
            )

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

    def init_redis(self, wait_retries: int = 60000):
        # For command fetch via redis connection
        self.redis_controller = RedisStreamHandler(
            ip=self.remote_ip, port=int(self.config.redis)
        )
        for _ in range(wait_retries):
            try:
                self.redis_controller.redis_client.ping()
                time.sleep(2)
                break
            except Exception:
                pass
        self.redis_controller.redis_client.ping()
        logger.debug(
            f"[{self.role}] Init redis at {self.remote_ip}:{self.redis_controller.port}"
        )

    def heartbeat_trigger(self, shutdown_signal: threading.Event):
        while True:
            if shutdown_signal.is_set():
                logger.info(
                    "[Policy] Heartbeat thread is stopped since the shutdown signal is set."
                )
                break
            max_retry = 3
            while max_retry > 0:
                try:
                    r = requests.post(
                        f"{self.remote_host}/api/heartbeat",
                        json={
                            "replica_name": self.replica_name,
                        },
                    )
                    if r.status_code != 200:
                        logger.error(
                            f"[Policy] Error response in send heartbeat: {r.json()}"
                        )
                    else:
                        break
                except Exception as e:
                    logger.error(
                        f"[Policy] Failed to send heartbeat to controller: {e}. Retrying..."
                    )
                finally:
                    max_retry -= 1
                    if max_retry == 0:
                        raise RuntimeError(
                            f"[Policy] Failed in in send heartbeat to controller after {max_retry} retries."
                        )
                    time.sleep(0.5)
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
