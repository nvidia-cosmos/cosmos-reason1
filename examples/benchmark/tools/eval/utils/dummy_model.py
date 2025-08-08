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

from types import SimpleNamespace

import attrs


@attrs.define(slots=False)
class DummyOutput:
    text = "C"


@attrs.define(slots=False)
class DummyRequestOutput:
    outputs = [DummyOutput()]  # noqa: RUF012


class DummyModel:
    def __init__(self):
        pass

    def generate(self, inputs, *args, **kwargs):
        return [DummyRequestOutput()] * len(inputs)


@attrs.define(slots=False)
class DummyTokenizer:
    def __init__(self):
        self.tokenizer = SimpleNamespace()
        self.tokenizer.bos_token_id = 0
        self.tokenizer.eos_token_id = 1

    def apply_chat_template(self, *args, **kwargs):
        return "C"


@attrs.define(slots=False)
class SamplingParams:
    def __init__(self, *args, **kwargs):
        pass
