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

import os
import torch

from cosmos_reason1.utils.parallelism import ParallelDims
from cosmos_reason1.policy.config import Config as CosmosConfig
from cosmos_reason1.comm.base import CommMixin
from cosmos_reason1.dispatcher.protocol import Role


class RolloutWorkerBase(CommMixin):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims) -> None:
        super().__init__()
        self.config = config
        self.role = Role.ROLLOUT
        self.parallel_dims = parallel_dims
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))  # rank in the node
        self.global_rank = int(os.environ.get("RANK", 0))  # rank in replica
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
