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
from cosmos_reason1.utils.constant import RedisStreamConstant
from typing import List


class RedisStreamHandler:
    def __init__(self, ip: str, port: int):
        """
        Initialize the RedisStreamHandler.

        Args:
            ip (str): The IP address of the Redis server.
            port (int): The port of the Redis server.
            stream_name (str): The name of the Redis stream to interact with.
        """
        self.ip = ip
        self.port = port
        self.redis_client = redis.Redis(
            host=self.ip, port=self.port, db=0, decode_responses=False
        )
        self.latest_id_command = "0-0"
        self.latest_id_rollout = "0-0"

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
        self.redis_client.xadd(
            stream_name + "_command",
            message,
            maxlen=RedisStreamConstant.STREAM_MAXLEN,
        )

    def subscribe_command(self, stream_name: str) -> List[dict]:
        """
        Read data from the Redis stream.

        Args:
            stream_name (str): The name of the Redis stream to read from.

        Returns:
            list: A list of stream entries.
        """
        messages = self.redis_client.xread(
            {stream_name + "_command": self.latest_id_command},
            count=RedisStreamConstant.CMD_FETCH_SIZE,
            block=RedisStreamConstant.CMD_READING_TIMEOUT_MS,
        )
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
        self.redis_client.xadd(
            stream_name + "_rollout",
            message,
            maxlen=RedisStreamConstant.STREAM_MAXLEN,
        )

    def subscribe_rollout(self, stream_name: str, count: int = -1) -> List[dict]:
        """
        Read data from the Redis stream.

        Args:
            stream_name (str): The name of the Redis stream to read from.
            count (int): The number of messages to read.

        Returns:
            list: A list of stream entries.
        """
        messages = self.redis_client.xread(
            {stream_name + "_rollout": self.latest_id_rollout},
            count=RedisStreamConstant.ROLLOUT_FETCH_SIZE if count <= 0 else count,
            block=RedisStreamConstant.ROLLOUT_READING_TIMEOUT_MS,
        )
        rollouts = []
        if messages:
            for _, message_list in messages:
                for message_id, message_data in message_list:
                    rollouts.append(message_data[b"rollout"])
                    self.latest_id_rollout = message_id
        return rollouts
