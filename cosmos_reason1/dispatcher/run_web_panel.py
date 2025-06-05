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
import toml
from fastapi import FastAPI
from contextlib import asynccontextmanager
from torch.utils.data import Dataset
import time
import asyncio
import base64
import cloudpickle

from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, List, Optional, Callable
from cosmos_reason1.dispatcher.controller import Controller
import cosmos_reason1.utils.constant as constant
from cosmos_reason1.dispatcher.protocol import MESH_NAMES
from cosmos_reason1.dispatcher.replica import Atom, RolloutGroup, Rollout, Replica
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
    NcclErrRequest,
)
from cosmos_reason1.policy.config import Config as CosmosConfig
import cosmos_reason1.utils.util as util
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.constant import COSMOS_ROLLOUT_SCAN_INTERVAL
from cosmos_reason1.utils.api_suffix import (
    COSMOS_API_PANEL_SUFFIX,
    COSMOS_API_STATUS_SUFFIX,
    COSMOS_API_META_SUFFIX,
    COSMOS_API_REGISTER_SUFFIX,
    COSMOS_API_SET_PROFILE_SUFFIX,
    COSMOS_API_SET_TRACE_PATH_SUFFIX,
    COSMOS_API_UNREGISTER_SUFFIX,
    COSMOS_API_HEARTBEAT_SUFFIX,
    COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX,
    COSMOS_API_NCCL_COMM_ACCEPTOR_SUFFIX,
    COSMOS_API_NCCL_COMM_GET_ALL_SUFFIX,
    COSMOS_API_NCCL_COMM_ERROR_SUFFIX,
    COSMOS_API_NEXT_PROMPT_SUFFIX,
    COSMOS_API_ROLLOUT_SUFFIX,
    COSMOS_API_POLICY_TRAIN_ACK_SUFFIX,
    COSMOS_API_POLICY_WEIGHT_READY_SUFFIX,
)
from cosmos_reason1.dispatcher.data.packer.base import DataPacker


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


@app.get(COSMOS_API_PANEL_SUFFIX)
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


@app.get(COSMOS_API_STATUS_SUFFIX)
async def get_status():
    return {
        "mesh_names": MESH_NAMES,
        "policy_replicas": _serialize_replicas(controller.policy_replicas),
        "rollout_replicas": _serialize_replicas(controller.rollout_replicas),
    }


@app.get(COSMOS_API_META_SUFFIX)
async def meta():
    meta = {
        "config": controller.config,
    }
    if not controller.is_rl and controller.sft_user_dataset is not None:
        meta["sft_user_dataset"] = base64.b64encode(
            cloudpickle.dumps(controller.sft_user_dataset)
        ).decode("utf-8")
    if controller.user_data_packer is not None:
        meta["user_data_packer"] = base64.b64encode(
            cloudpickle.dumps(controller.user_data_packer)
        ).decode("utf-8")
    return meta


@app.post(COSMOS_API_REGISTER_SUFFIX)
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


@app.post(COSMOS_API_SET_PROFILE_SUFFIX)
async def set_profile(request: SetProfileRequest):
    logger.info(f"[Dispatcher] set profile request: {request}")
    msg = await controller.set_profile(request)
    return msg


@app.post(COSMOS_API_SET_TRACE_PATH_SUFFIX)
async def set_trace_path(request: SetTracePathRequest):
    atom = await controller.set_trace_path(
        request.replica_name, request.trace_path, request.global_rank
    )
    if atom is not None:
        return {"message": f"Trace path set for atom: {atom}"}
    else:
        return {"message": "Ignore the trace path request!"}


@app.post(COSMOS_API_UNREGISTER_SUFFIX)
async def unregister(request: UnregisterRequest):
    await controller.unregister(request.replica_name)
    return {"message": "Unregistered"}


@app.post(COSMOS_API_HEARTBEAT_SUFFIX)
async def heartbeat(request: HeartbeatRequest):
    # Set the replica timestamp to the current time for heartbeat
    controller.set_replica_timestamp(request.replica_name, int(time.time()))
    return {"message": "Heartbeat received"}


"""
NCCL Handshake API
"""


@app.post(COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX)
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


@app.post(COSMOS_API_NCCL_COMM_ACCEPTOR_SUFFIX)
async def comm_acceptor(request: HandshakeAcceptorRequest):
    if request.unique_pair_name not in controller.temp_kv_store:
        return create_error_response(
            constant.ErrorCode.INTERNAL_ERROR, "Unique pair name not found"
        )
    return {"handle_base64": controller.temp_kv_store.get(request.unique_pair_name)}


@app.post(COSMOS_API_NCCL_COMM_ERROR_SUFFIX)
async def comm_error(request: NcclErrRequest):
    await controller.set_replica_ncclerror(request.replica_name, request.error)
    return {"message": "DetectTimeout received"}


@app.get(COSMOS_API_NCCL_COMM_GET_ALL_SUFFIX)
async def comm_get_all():
    return {"comm_info": controller.temp_kv_store}


"""
Rollout API
"""


@app.get(COSMOS_API_NEXT_PROMPT_SUFFIX)
async def get_batched_prompt(n: int):
    prompt_id_and_payload_list, is_end = await controller.get_batched_prompt(n)
    return {
        "prompt_id_and_payload_list": prompt_id_and_payload_list,
        "is_end": is_end,
    }


async def monitor_replica_status():
    while True:
        now = time.time()
        await controller.maintain_replica_life_status(now)
        await asyncio.sleep(COSMOS_ROLLOUT_SCAN_INTERVAL)


@app.post(COSMOS_API_ROLLOUT_SUFFIX)
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
                payload=payload,
                completions=completions,
                extra_info=rollout.extra_info,
                reference_answer=controller.query_reference_answer(prompt_idx),
            )
            for prompt_idx, payload, completions in zip(
                rollout.prompt_idxs, rollout.payloads, rollout.completions
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


@app.post(COSMOS_API_POLICY_TRAIN_ACK_SUFFIX)
async def train_ack(request: TrainAckRequest):
    try:
        replicaname = request.replica_name
        iteration_count = request.iteration_count
        profile_finished = request.profile_finished
        await controller.train_ack(replicaname, iteration_count, profile_finished)
        return {"message": "Ack completed"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


@app.post(COSMOS_API_POLICY_WEIGHT_READY_SUFFIX)
async def weight_ready(request: WeightReadyRequest):
    try:
        replicaname = request.replica_name
        await controller.weight_ready(replicaname)
        return {"message": "Weight ready received"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


def _serialize_replicas(replicas: Dict[str, Replica]) -> List[Dict]:
    result = []
    for name, replica in replicas.items():
        result.append(replica.to_dict())
    return result


def main(
    dataset: Optional[Dataset] = None,
    data_packer: Optional[DataPacker] = None,
    reward_fns: Optional[List[Callable]] = None,
):
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

        if data_packer is not None:
            assert isinstance(
                data_packer, DataPacker
            ), "data_packer should be a DataPacker instance"
        controller.setup(
            loaded_config,
            redis_port=args.redis_port,
            redis_logfile_path=args.redis_logfile_path,
            dataset=dataset,
            reward_fns=reward_fns,
            data_packer=data_packer,
        )
        logger.info(f"Successfully loaded configuration from {args.config_file}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {args.config_file}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load or parse config file {args.config_file}: {e}.",
            exc_info=True,
        )

    uvicorn.run(
        app, host="0.0.0.0", port=util.find_available_port(args.port), access_log=False
    )


if __name__ == "__main__":
    main()
