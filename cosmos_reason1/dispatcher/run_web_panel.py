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

import argparse
import uvicorn
import os
import uuid
import toml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, List, Optional
from cosmos_reason1.dispatcher.controller import Controller
import cosmos_reason1.utils.constant as constant
from cosmos_reason1.dispatcher.protocol import MESH_NAMES
from cosmos_reason1.dispatcher.replica import Atom, RolloutGroup, Rollout
from cosmos_reason1.dispatcher.protocol import (
    RegisterRequest,
    ErrorResponse,
    RolloutRequest,
    HandshakeInitiatorRequest,
    HandshakeAcceptorRequest,
    UnregisterRequest,
    TrainAckRequest,
    HeartbeatRequest,
    WeightReadyRequest,
    SetProfileRequest,
    SetTracePathRequest,
)
from cosmos_reason1.policy.config import Config as CosmosConfig
import cosmos_reason1.utils.util as util
import subprocess
import sys
import atexit
from cosmos_reason1.utils.logging import logger
import time
from cosmos_reason1.utils.constant import COSMOS_ROLLOUT_SCAN_INTERVAL
import asyncio
from contextlib import asynccontextmanager
from cosmos_reason1.utils.network_util import get_eth_ips
from cosmos_reason1.utils.modelscope import update_config_if_modelscope


def create_error_response(
    code: int, message: str, status_code: Optional[int] = None
) -> JSONResponse:
    if status_code is None:
        status_code = code // 100
    return JSONResponse(
        ErrorResponse(message=message, code=code).model_dump(), status_code=status_code
    )


controller = Controller()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if controller.config.train.train_policy.type != "sft":
        asyncio.create_task(monitor_replica_status())
    yield


app = FastAPI(lifespan=lifespan)


REDIS_CONFIG_PATH = "/opt/redis_config.conf"


def write_redis_config(file_path, port, logfile):
    config_content = f"""# Redis configuration file example for insecure connections

# Bind to all network interfaces (use with caution)
bind 0.0.0.0

# Set the port for Redis to listen on (default is {port})
port {port}

# Disable TLS by setting the tls-port to 0
tls-port 0

# Disable authentication by commenting out the requirepass directive
# requirepass yourpassword

# Other configuration settings can remain as default or be customized as needed
timeout 0
tcp-keepalive 300
protected-mode no
# enable-protected-configs yes
# enable-debug-command yes
# enable-module-command yes
daemonize yes
supervised no
loglevel notice
logfile {logfile}
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /opt
"""
    with open(file_path, "w") as file:
        file.write(config_content)


@app.get("/panel")
async def panel():
    # HTML template with JavaScript for auto-refresh
    with open(
        os.path.join(
            os.path.dirname(__file__), "config/frontend", "dispatcher_status.html"
        ),
        "r",
        encoding="utf-8",
    ) as file:
        html = file.read()
    return HTMLResponse(html)


"""
API for replica-controller communication
"""


@app.get("/api/status")
async def get_status():
    return {
        "mesh_names": MESH_NAMES,
        "policy_replicas": _serialize_replicas(controller.policy_replicas),
        "rollout_replicas": _serialize_replicas(controller.rollout_replicas),
    }


@app.get("/api/meta")
async def get_meta():
    return {
        "config": controller.config,
    }


@app.post("/api/register")
async def register(request: RegisterRequest):
    try:
        await controller.register(
            Atom.from_register_request(request), role=request.role
        )
        return {"message": "Registered"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


@app.post("/api/set_profile")
async def set_profile(request: SetProfileRequest):
    msg = await controller.set_profile(request.replica_name)
    return msg


@app.post("/api/set_trace_path")
async def set_trace_path(request: SetTracePathRequest):
    atom = await controller.set_trace_path(
        request.replica_name, request.trace_path, request.global_rank
    )
    if atom is not None:
        return {"message": f"Trace path set for atom: {atom}"}
    else:
        return {"message": "Ignore the trace path request!"}


@app.post("/api/unregister")
async def unregister(request: UnregisterRequest):
    await controller.unregister(request.replica_name)
    return {"message": "Unregistered"}


@app.post("/api/heartbeat")
async def heartbeat(request: HeartbeatRequest):
    # Set the replica timestamp to the current time for heartbeat
    controller.set_replica_timestamp(request.replica_name, int(time.time()))
    return {"message": "Heartbeat received"}


"""
NCCL Handshake API
"""


@app.post("/api/nccl/comm_initiator")
async def comm_initiator(request: HandshakeInitiatorRequest):
    if request.unique_pair_name in controller.temp_kv_store:
        return create_error_response(
            constant.ErrorCode.ALREADY_EXISTS, "Unique pair name already exists"
        )
    elif request.handle_base64 is None or request.handle_base64 == "":
        return create_error_response(
            constant.ErrorCode.INVALID_REQUEST, "Handle is required"
        )

    await controller.update_kv_store(request.unique_pair_name, request.handle_base64)
    return {"message": "Handshake initiator received"}


@app.post("/api/nccl/comm_acceptor")
async def comm_acceptor(request: HandshakeAcceptorRequest):
    if request.unique_pair_name not in controller.temp_kv_store:
        return create_error_response(
            constant.ErrorCode.INTERNAL_ERROR, "Unique pair name not found"
        )
    return {"handle_base64": controller.temp_kv_store.get(request.unique_pair_name)}


"""
Rollout API
"""


@app.get("/api/next_prompt")
async def get_batched_prompt(n: int):
    prompt_id_and_str_list, is_end = await controller.get_batched_prompt(n)
    return {
        "prompt_id_and_str_list": prompt_id_and_str_list,
        "is_end": is_end,
    }


async def monitor_replica_status():
    while True:
        now = time.time()
        await controller.maintain_replica_life_status(now)
        await asyncio.sleep(COSMOS_ROLLOUT_SCAN_INTERVAL)


@app.post("/api/rollout")
async def put_rollout(rollout: RolloutRequest):
    try:
        if rollout.extra_info is not None and "is_end" in rollout.extra_info:
            # If the extra info contains "is_end", it means the rollout is finished
            await controller.handle_rollout_end_ack(
                rollout.extra_info, rollout.src_replica_name
            )
            return {"message": "Rollout end put"}

        rollout_groups: List[RolloutGroup] = [
            RolloutGroup(
                prompt_idx=prompt_idx,
                prompt=prompt,
                completions=completions,
                extra_info=rollout.extra_info,
                reference_answer=controller.query_reference_answer(prompt_idx),
            )
            for prompt_idx, prompt, completions in zip(
                rollout.prompt_idxs, rollout.prompt_strs, rollout.completions
            )
        ]

        rollouts_list: List[List[Rollout]] = [
            rollout_group.compute_rollouts(controller.rl_algo)
            for rollout_group in rollout_groups
        ]

        # Dynamic Sampling: Filter out the rollouts that the rewards are all the same
        valid_rollouts_list: List[List[Rollout]] = []
        invalid_rollouts_list: List[List[Rollout]] = []
        for rollouts_group in rollouts_list:
            if len(set([rollout.reward for rollout in rollouts_group])) > 1:
                valid_rollouts_list.append(rollouts_group)
            else:
                # If the rewards are all the same, we need to sample one rollout from the group
                invalid_rollouts_list.append(rollouts_group)

        # Flatten the rollouts into a single list
        valid_rollouts = [
            rollout
            for rollouts_group in valid_rollouts_list
            for rollout in rollouts_group
        ]
        invalid_rollouts = [
            rollout
            for rollouts_group in invalid_rollouts_list
            for rollout in rollouts_group
        ]

        if len(valid_rollouts) > 0:
            logger.debug(
                f"[RolloutGroup] from replica: {rollout.src_replica_name} with {len(rollout.completions)} samples:"
                f"example: rollouts[0]\n{valid_rollouts[0]}"
            )

        await controller.put_rollouts(valid_rollouts, invalid_rollouts)
        return {"message": "Rollout put"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


@app.get("/api/policy/{replica_name}/rollouts")
async def get_rollouts(replica_name: str):
    try:
        rollouts = await controller.get_rollouts(replica_name, 0)
        return {"message": "Rollouts fetched", "rollouts": rollouts}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


@app.post("/api/policy/train_ack")
async def train_ack(request: TrainAckRequest):
    try:
        replicaname = request.replica_name
        iteration_count = request.iteration_count
        await controller.train_ack(replicaname, iteration_count)
        return {"message": "Ack completed"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


@app.post("/api/policy/weight_ready")
async def weight_ready(request: WeightReadyRequest):
    try:
        replicaname = request.replica_name
        await controller.weight_ready(replicaname)
        return {"message": "Weight ready received"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


def _serialize_replicas(replicas: Dict) -> List[Dict]:
    result = []
    for name, replica in replicas.items():
        atoms = []
        for atom_key, atom in replica.atoms.items():
            atoms.append(
                {
                    "ranks": atom.ranks,
                    "group_size": atom.group_size,
                    "replica_name": atom.replica_name,
                }
            )
        result.append(
            {
                "name": replica.name,
                "role": replica.role,
                "atoms": atoms,
                "arrived": replica.all_atoms_arrived,
                "weight_step": replica.weight_step,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the web panel for the dispatcher."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the web panel on."
    )
    parser.add_argument(
        "--redis-port", type=int, default=12800, help="Port to run the web panel on."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        required=True,
        help="Path to TOML configuration file to load.",
    )
    parser.add_argument(
        "--create-redis-config",
        action="store_true",
        help="Whether to create the default redis config that allows insecure connections.",
    )
    parser.add_argument(
        "--redis-logfile-path",
        type=str,
        default="/opt/redis.log",
        help="The redis server log file path.",
    )
    args = parser.parse_args()

    # Load config from file if provided
    loaded_config = None
    assert os.path.exists(
        args.config_file
    ), f"Config file {args.config_file} does not exist."

    try:
        logger.info(f"Attempting to load configuration from {args.config_file}")
        with open(args.config_file, "r") as f:
            config_dict = toml.load(f)

        # Ensure CosmosConfig is available (it's imported at the top now)
        # from cosmos_reason1.policy.config import Config as CosmosConfig
        # Need SFTDataConfig and GrpoConfig for from_dict

        loaded_config = CosmosConfig.from_dict(config_dict)
        loaded_config = update_config_if_modelscope(loaded_config)
        # Use redis port from config if available, otherwise use arg/default
        if hasattr(loaded_config, "redis") and loaded_config.redis:
            try:
                redis_port_from_config = int(loaded_config.redis)
                args.redis_port = redis_port_from_config
                logger.info(f"Using Redis port {args.redis_port} from config file.")
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid redis port format in config file: {loaded_config.redis}. Using default/arg: {args.redis_port}"
                )
        controller.set_config(loaded_config)
        logger.info(f"Successfully loaded configuration from {args.config_file}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {args.config_file}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load or parse config file {args.config_file}: {e}.",
            exc_info=True,
        )

    redis_free_port = util.find_available_port(args.redis_port)
    args.redis_port = redis_free_port  # Update args with the actual port found

    # Update the redis port in the loaded config object if it exists
    if controller.config:  # Check if config was set (either loaded or via web later)
        controller.config.redis = str(redis_free_port)

    ips = get_eth_ips()
    if len(ips) > 0:
        controller.config.eth_ips = ";".join(ips)

    random_db_file_name = f"cosmos_reason1_{str(uuid.uuid4())}.rdb"
    if args.create_redis_config:
        write_redis_config(REDIS_CONFIG_PATH, redis_free_port, args.redis_logfile_path)
        redis_server_cmd = f'redis-server {REDIS_CONFIG_PATH} --dbfilename {random_db_file_name} --save ""'
    else:
        redis_server_cmd = f'redis-server --port {redis_free_port} --dbfilename {random_db_file_name} --save ""'
    logger.info(f"Init redis {redis_server_cmd}.")
    redis_server_proc = subprocess.Popen(
        redis_server_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr
    )
    controller.init_redis(host="0.0.0.0", port=redis_free_port)
    uvicorn.run(
        app, host="0.0.0.0", port=util.find_available_port(args.port), access_log=False
    )

    def exit_server():
        logger.info("Stopping redis server")
        redis_server_proc.terminate()
        redis_server_proc.wait()
        if args.create_redis_config:
            redis_terminate_cmd = f"redis-cli -p {redis_free_port} shutdown nosave"
            redis_terminate = subprocess.Popen(
                redis_terminate_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr
            )
            redis_terminate.wait()
        logger.info("Redis server stopped.")

    atexit.register(exit_server)


if __name__ == "__main__":
    main()
