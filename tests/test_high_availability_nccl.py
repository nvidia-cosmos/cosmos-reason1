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
import time
import os
import sys
import subprocess
import torch.distributed as dist
import threading
from functools import partial
from typing import override
import requests
import atexit
from queue import Queue, Empty

from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils import constant
from cosmos_reason1.utils.network_util import make_request_with_retry, get_local_ip
from cosmos_reason1.utils.distributed import HighAvailabilitylNccl
from cosmos_reason1.dispatcher.command import Command, BuildMeshCommand
from cosmos_reason1.policy.config import Config
from cosmos_reason1.utils.distributed import get_controller_metadata
from cosmos_reason1.dispatcher.protocol import MESH_NAMES
from cosmos_reason1.comm.base import CommMixin
from cosmos_reason1.utils.parallelism import ParallelDims
from cosmos_reason1.utils.api_suffix import (
    COSMOS_API_REGISTER_SUFFIX,
    COSMOS_API_UNREGISTER_SUFFIX,
    COSMOS_API_POLICY_WEIGHT_READY_SUFFIX,
)


WORK_DIR = f"/tmp/{os.path.basename(__file__)}"
CTRL_PORT = 8010


os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{CTRL_PORT}"
os.environ["COSMOS_NCCL_TIMEOUT_MS"] = "30000"
os.environ["COSMOS_LOG_LEVEL"] = "DEBUG"

# os.environ["NCCL_DEBUG"] = "INFO"


def write_train_config():
    os.makedirs(WORK_DIR, exist_ok=True)

    config = f"""
redis = "12808"

[train]
resume = "False"
epoch = 1
output_dir = "{WORK_DIR}"

[logging]
enable_logging = false
project_name = "cosmos_reason1"
experiment_name = "None"

[train.train_policy]
type = "grpo"
dataset_name = "JiaxinTsao/math_examples"
prompt_column_name = "prompt"
response_column_name = "result"
dataset_train_split = "train"
reward_function = "boxed_math"
temperature = 0.9
epsilon_low = 0.2
epsilon_high = 0.2
kl_beta = 0.0
mu_iterations = 1

[policy.parallelism]
n_init_replicas = 1
tp_size = 1
cp_size = 1
dp_shard_size = 1
pp_size = 1
dp_replicate_size = 1
"""
    cfg_file_path = os.path.join(WORK_DIR, "train_config.toml")
    with open(cfg_file_path, "w") as f:
        f.write(config)
    return cfg_file_path


def launch_controller(config: str):
    """
    Launch the controller process.
    """
    logger.info("launch controller")
    p = subprocess.Popen(
        "python -m cosmos_reason1.dispatcher.run_web_panel "
        f"--port {CTRL_PORT} --config {config}",
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        env=dict(os.environ),
    )

    return [p]


class TestHANccl(CommMixin):
    def __init__(self, ctrl_ip: str, ctrl_port: int, config: Config):
        self.config = config
        self.replica_name = f"HANCCL-{dist.get_rank()}"
        self.global_rank = (
            0  # here we assume every replica only has one gpu, so the global_rank is 0
        )
        self.replica_rank = dist.get_rank()
        self.remote_hosts = [f"http://{os.environ["COSMOS_CONTROLLER_HOST"]}"]
        self.role = "POLICY"
        self._shutdown_event = threading.Event()
        self._is_registered = False

        # override those fields
        self.remote_ips = ctrl_ip
        self.remote_port = ctrl_port
        self.parallel_dims = ParallelDims.from_config_for_analysis(
            config.policy.parallelism, world_size=1
        )
        self.parallel_dims.build_mesh(device_type="cuda")

        # init redis
        self.init_redis()

        # start heartbeat thread
        self.heartbeat_thread = self.start_heartbeat(
            self._shutdown_event,
        )

        self.fetch_command_thread = threading.Thread(
            target=self.run_fetch_command,
            daemon=True,
        )
        self.fetch_command_thread.start()
        self.command_queue = Queue()

    def __get_alternative_urls(self, suffix: str):
        # Get the alternative URLs for the given suffix
        urls = []
        for remote_host in self.remote_hosts:
            urls.append(f"{remote_host}/{suffix}")
        return urls

    @override
    def register_to_controller(self):
        if self._is_registered:
            return

        host_info_tuple = get_local_ip()
        if host_info_tuple is None:
            raise Exception("Failed to get local IP address")
        host_ip, host_name = host_info_tuple
        make_request_with_retry(
            partial(
                requests.post,
                json={
                    "replica_name": self.replica_name,
                    "role": self.role,
                    "mesh_names": MESH_NAMES,
                    "ranks": [0 for _ in MESH_NAMES],
                    "group_size": [1 for _ in MESH_NAMES],
                    "global_rank": self.global_rank,
                    "host_ip": host_ip,
                    "host_name": host_name,
                },
            ),
            self.__get_alternative_urls(COSMOS_API_REGISTER_SUFFIX),
            max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
        )
        self._is_registered = True
        logger.info(f"register to controller: {self.replica_name}")

        if self.replica_rank == 0:
            # MOCK, tell the controller the weight is ready, then buildmesh and weight sync command will be triggered
            make_request_with_retry(
                partial(
                    requests.post,
                    json={"replica_name": self.replica_name},
                ),
                self.__get_alternative_urls(COSMOS_API_POLICY_WEIGHT_READY_SUFFIX),
            )

    @override
    def unregister_from_controller(self):
        if not self._is_registered:
            return

        make_request_with_retry(
            partial(
                requests.post,
                json={"replica_name": self.replica_name},
            ),
            self.__get_alternative_urls(COSMOS_API_UNREGISTER_SUFFIX),
            max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
        )
        self._is_registered = False
        logger.info(f"unregister from controller: {self.replica_name}")

    def run_fetch_command(self):
        while not self._shutdown_event.is_set():
            if not self._is_registered:
                time.sleep(0.1)
                continue
            raw_cmds = self.redis_controller.subscribe_command(self.replica_name)

            for cmd in raw_cmds:
                self.command_queue.put(Command.depack(cmd))
            time.sleep(0.1)
        logger.info(f"fetch command thread is stopped: {self.replica_name}")

    def fetch_command(self, block: bool = True) -> list[Command]:
        if not self._is_registered:
            return []

        try:
            cmds = [self.command_queue.get(block=block)]
            while not self.command_queue.empty():
                cmds.append(self.command_queue.get())
        except Empty:
            return []
        return cmds

    def test_comm_auto_rebuild_normal(self):
        logger.info(f"run test_comm_auto_rebuild {self.replica_name}")

        # 0. register to controller, let controller trigger the buildmesh command
        self.register_to_controller()

        # 1. create a comm
        comm = HighAvailabilitylNccl(
            replica_name=self.replica_name,
            global_rank=self.global_rank,
            controller_hosts=self.remote_hosts,
        )

        # wait all ranks fetch latest command from controller
        dist.barrier()
        time.sleep(2)

        query_cmd = self.fetch_command()
        for cmd in query_cmd:
            if isinstance(cmd, BuildMeshCommand):
                logger.info(
                    f"  ({self.replica_name}) push build mesh command: {cmd.replica_name_to_rank}"
                )
                comm.push_cmd(cmd)

        # 2. wait for the comm to be ready
        comm.wait_comm_ready()
        assert (
            comm.world_size() == dist.get_world_size()
        ), f"world size should be {dist.get_world_size()}"
        comm.destroy_nccl_comm()
        comm.shutdown()
        logger.info("  === normal case, passed")

    def test_comm_auto_rebuild_intiative_scale_down(self):
        logger.info(
            f"run test_comm_auto_rebuild_intiative_scale_down {self.replica_name}"
        )

        # 0. register to controller, let controller trigger the buildmesh command
        self.register_to_controller()

        # 1. create a comm
        comm = HighAvailabilitylNccl(
            replica_name=self.replica_name,
            global_rank=self.global_rank,
            controller_hosts=self.remote_hosts,
        )

        # wait all ranks fetch latest command from controller
        dist.barrier()
        time.sleep(2)

        # 2. test intiative scale down
        if self.replica_rank == 1:
            # use unregister to trigger the buildmesh command
            self.unregister_from_controller()
        else:
            # other rank do scale down buildmesh
            while True:
                # wait until controller trigger the buildmesh command
                cmds = self.fetch_command()
                cmds = [cmd for cmd in cmds if isinstance(cmd, BuildMeshCommand)]
                if len(cmds) == 0:
                    time.sleep(0.1)
                    continue
                for cmd in cmds:
                    comm.push_cmd(cmd)
                break
            comm.wait_comm_ready()
            assert (
                comm.world_size() == dist.get_world_size() - 1
            ), f"world size should be {dist.get_world_size() - 1}, actual {comm.world_size()}"

        comm.destroy_nccl_comm()
        comm.shutdown()
        logger.info("  === intiative scale down, passed")

    def test_comm_auto_rebuild_timeout_scale_down(self):
        logger.info(
            f"run test_comm_auto_rebuild_intiative_scale_down_and_scale_up {self.replica_name}"
        )

        # 0. register to controller, let controller trigger the buildmesh command
        self.register_to_controller()

        # 1. create a comm
        comm = HighAvailabilitylNccl(
            replica_name=self.replica_name,
            global_rank=self.global_rank,
            controller_hosts=self.remote_hosts,
        )

        # 2. trigger buildmesh command over all ranks
        if self.replica_rank == 1:
            self.unregister_from_controller()
            self.register_to_controller()
            logger.info("  re-register to controller is done")

        # wait all ranks fetch latest command from controller
        dist.barrier()
        time.sleep(5)

        cmds = self.fetch_command()
        cmds = [cmd for cmd in cmds if isinstance(cmd, BuildMeshCommand)]
        assert (
            len(cmds) > 0
        ), f"  replica_rank {self.replica_rank} should have at least one buildmesh command"

        # 3. test build mesh timeout
        # let rank 1 exit, so that other ranks will timeout, and rebuild mesh in scale down mode
        logger.info(
            f"  replica_rank {self.replica_rank} prepare buildmesh timeout environment"
        )
        if self.replica_rank == 1:
            logger.info(
                "  replica_rank 1 exit, wait for other ranks to timeout and execute the build mesh command"
            )
        else:
            # monitor that all other ranks execute the build mesh command
            cmd = cmds[-1]
            assert (
                len(cmd.replica_name_to_rank) == dist.get_world_size()
            ), f"buildmesh command should be {dist.get_world_size()}"
            logger.info(
                f"  replica_rank {self.replica_rank} push the buildmesh command: {cmd.replica_name_to_rank}"
            )
            comm.push_cmd(cmd)

            retry_count = 0
            while retry_count < 10:
                # here we wait rebuild comm until nranks equals to the world size - 1
                if comm.is_ready():
                    break

                # query timeout commands and push to comm,
                cmds = self.fetch_command(block=False)
                cmds = [cmd for cmd in cmds if isinstance(cmd, BuildMeshCommand)]
                for cmd in cmds:
                    logger.info(
                        f"  replica_rank {self.replica_rank} retry, push cmd: {cmd.replica_name_to_rank}"
                    )
                    comm.push_cmd(cmd)

                time.sleep(5)
                retry_count += 1

            assert comm.is_ready(), "comm is not ready"
            assert (
                comm.world_size() == dist.get_world_size() - 1
            ), f"world size should be {dist.get_world_size() - 1}, actual {comm.world_size()}"

        # finally, shutdown the comm
        comm.destroy_nccl_comm()
        comm.shutdown()
        logger.info(" === test_comm_auto_rebuild passed")

    def test_allreduce_timeout_retry(self):
        """
        Run the NCCL test.
        """
        logger.info("run nccl test")
        pass

    def test_send_recv_timeout_retry():
        """
        Run the NCCL test.
        """
        logger.info("run nccl test")
        pass


def cleanup():
    """
    Cleanup function to kill all processes.
    """
    # try kill created processes
    login_user = "root"
    try:
        login_user = os.getlogin()
    except Exception:
        pass
    subprocess.run(f"pkill -u {login_user} -f 'redis'", shell=True)
    subprocess.run(f"pkill -u {login_user} -f 'dispatcher.run_web_panel'", shell=True)
    # subprocess.run(f"pkill -u {login_user} -f 'torchrun'", shell=True)


atexit.register(cleanup)


def main():
    # 1. init process group in this node
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo")
    rank = dist.get_rank()

    # 2. only rank 0 launch the controller
    ctrl_hdl = []
    if rank == 0:
        cfg_path = write_train_config()
        ctrl_hdl.extend(launch_controller(cfg_path))

    # wait for controller to be ready
    time.sleep(10)

    # 3. init the tester
    ctrl_ip, ctrl_port, metadata = get_controller_metadata()
    cosmos_config = Config.from_dict(metadata["config"])
    tester = TestHANccl(ctrl_ip, ctrl_port, cosmos_config)
    dist.barrier()

    # 4. Run the test
    tester.test_comm_auto_rebuild_normal()
    dist.barrier()

    tester.test_comm_auto_rebuild_intiative_scale_down()
    dist.barrier()

    tester.test_comm_auto_rebuild_timeout_scale_down()
    dist.barrier()

    # finally, wait for all ranks to finish the test
    for hdl in ctrl_hdl:
        hdl.kill()
    dist.destroy_process_group()


if __name__ == "__main__":
    # let run this test in distributed mode
    if os.environ.get("LOCAL_RANK") is None:
        # n_gpu = torch.cuda.device_count()
        n_gpu = 4
        command = [
            "torchrun",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(n_gpu),
            os.path.abspath(__file__),
        ]
        subprocess.run(command)
    else:
        main()
