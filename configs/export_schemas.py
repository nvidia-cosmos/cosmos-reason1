#!/usr/bin/env -S uv run --script
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

#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cosmos-reason1-utils",
#   "msgspec",
#   "vllm",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# [tool.uv.sources]
# cosmos-reason1-utils = {path = "../cosmos_reason1_utils", editable = true}
# ///

"""Export config schemas."""

import argparse
import json
import pathlib

import msgspec
import vllm

from cosmos_reason1_utils.vision import VisionConfig

SCRIPT = pathlib.Path(__file__).resolve()


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"{SCRIPT.parent}/schemas",
        help="Output directory",
    )
    args = args.parse_args()

    output_dir = pathlib.Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_schema = VisionConfig.model_json_schema()
    (output_dir / "vision_config.json").write_text(json.dumps(vision_schema, indent=2))

    sampling_params = msgspec.json.schema(vllm.SamplingParams)
    (output_dir / "sampling_params.json").write_bytes(
        msgspec.json.format(msgspec.json.encode(sampling_params), indent=2)
    )


if __name__ == "__main__":
    main()
