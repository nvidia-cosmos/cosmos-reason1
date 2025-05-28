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

import redis
from datetime import datetime
from cosmos_reason1.utils.constant import (
    RedisStreamConstant,
    COSMOS_HTTP_RETRY_CONFIG,
    COSMOS_HTTP_LONG_WAIT_MAX_RETRY,
)
from typing import List
from cosmos_reason1.utils.util import make_request_with_retry
from functools import partial
from cosmos_reason1.utils.logging import logger
import enum


class RedisOpType(enum.Enum):
    XADD = "add"
    XREAD = "read"
    PING = "ping"


class RedisStreamHandler:
    def __init__(self, ips: List[str], port: int):
        """
        Initialize the RedisStreamHandler.

        Args:
            ips (List[str]): The alternative IP addresses of the Redis server.
            port (int): The port of the Redis server.
            stream_name (str): The name of the Redis stream to interact with.
        """
        self.ips = ips
        self.port = port
        self.redis_clients = []
        for ip in ips:
            self.redis_clients.append(
                redis.Redis(host=ip, port=self.port, db=0, decode_responses=False)
            )
        self.latest_id_command = "0-0"
        self.latest_id_rollout = "0-0"
        self.ping()

    def publish_command(self, data, stream_name: str):
        """
        Write data to the Redis stream.

        Args:
            data : The packed command to write to the stream.

        Returns:
            str: The ID of the added stream entry.
        """
        message = {"command": data, "timestamp": datetime.now().isoformat()}
        # Add message to stream
        try:
            make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.XADD,
                    stream_name + "_command",
                    message,
                    maxlen=RedisStreamConstant.STREAM_MAXLEN,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to write to Redis stream {stream_name}_command: {e}")
            raise e

    def subscribe_command(self, stream_name: str) -> List[dict]:
        """
        Read data from the Redis stream.

        Args:
            stream_name (str): The name of the Redis stream to read from.

        Returns:
            list: A list of stream entries.
        """
        try:
            messages = make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.XREAD,
                    {stream_name + "_command": self.latest_id_command},
                    count=RedisStreamConstant.CMD_FETCH_SIZE,
                    block=RedisStreamConstant.CMD_READING_TIMEOUT_MS,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to read from Redis stream {stream_name}_command: {e}")
            raise e
        commands = []
        if messages:
            for _, message_list in messages:
                for message_id, message_data in message_list:
                    commands.append(message_data[b"command"])
                    self.latest_id_command = message_id
        return commands

    def publish_rollout(self, data, stream_name: str):
        """
        Write data to the Redis stream.

        Args:
            data : The packed rollout to write to the stream.

        Returns:
            str: The ID of the added stream entry.
        """
        message = {"rollout": data, "timestamp": datetime.now().isoformat()}
        # Add message to stream
        try:
            make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.XADD,
                    stream_name + "_rollout",
                    message,
                    maxlen=RedisStreamConstant.STREAM_MAXLEN,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to write to Redis stream {stream_name}_rollout: {e}")
            raise e

    def subscribe_rollout(self, stream_name: str, count: int = -1) -> List[dict]:
        """
        Read data from the Redis stream.

        Args:
            stream_name (str): The name of the Redis stream to read from.
            count (int): The number of messages to read.

        Returns:
            list: A list of stream entries.
        """
        try:
            messages = make_request_with_retry(
                self.requests_for_alternative_clients(
                    RedisOpType.XREAD,
                    {stream_name + "_rollout": self.latest_id_rollout},
                    count=RedisStreamConstant.ROLLOUT_FETCH_SIZE
                    if count <= 0
                    else count,
                    block=RedisStreamConstant.ROLLOUT_READING_TIMEOUT_MS,
                ),
                response_parser=None,
                max_retries=COSMOS_HTTP_RETRY_CONFIG.max_retries,
            )
        except Exception as e:
            logger.error(f"Failed to read from Redis stream {stream_name}_rollout: {e}")
            raise e
        rollouts = []
        if messages:
            for _, message_list in messages:
                for message_id, message_data in message_list:
                    rollouts.append(message_data[b"rollout"])
                    self.latest_id_rollout = message_id
        return rollouts

    def requests_for_alternative_clients(self, op: RedisOpType, *args, **kwargs):
        """
        Make requests to alternative clients based on the operation type.

        Args:
            op (RedisOpType): The operation type (XADD or XREAD or PING).
            *args: Positional arguments for the request.
            **kwargs: Keyword arguments for the request.

        Returns:
            list: A list of Callable objects for the requests.
        """
        calls = []
        if op == RedisOpType.XADD:
            for redis_client in self.redis_clients:
                calls.append(
                    partial(
                        redis_client.xadd,
                        *args,
                        **kwargs,
                    )
                )
        elif op == RedisOpType.XREAD:
            for redis_client in self.redis_clients:
                calls.append(
                    partial(
                        redis_client.xread,
                        *args,
                        **kwargs,
                    )
                )
        elif op == RedisOpType.PING:
            for redis_client in self.redis_clients:
                calls.append(redis_client.ping)
        else:
            raise ValueError(f"Unsupported operation type: {op}")
        return calls

    def ping(self):
        # Wait for redis to be ready
        try:
            make_request_with_retry(
                self.requests_for_alternative_clients(RedisOpType.PING),
                response_parser=None,
                max_retries=COSMOS_HTTP_LONG_WAIT_MAX_RETRY,
            )
        except Exception as e:
            logger.error(f"Failed to ping Redis when init Redis: {e}")
            raise e
