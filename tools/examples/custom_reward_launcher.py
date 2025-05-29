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


import random
from typing import Union
from cosmos_reason1.dispatcher.algo.reward import register_reward_fn
from cosmos_reason1.dispatcher.run_web_panel import main as launch_dispatcher

def custom_reward_fn(to_be_evaluated: str, reference: Union[str, None], *args, **kwargs) -> float:
    return random.random()

register_reward_fn("custom_reward_fn", custom_reward_fn, override_toml=True)

if __name__ == "__main__":
    launch_dispatcher()