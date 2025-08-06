#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "msgspec",
#   "pydantic",
#   "qwen-vl-utils",
#   "torch",
#   "torchvision",
#   "vllm",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# [tool.uv.sources]
# qwen-vl-utils = {git = "https://github.com/spectralflight/Qwen2.5-VL.git", branch = "cosmos", subdirectory = "qwen-vl-utils"}
# ///

"""Export config schemas."""

import argparse
import json
import pathlib
import pydantic
import qwen_vl_utils
import msgspec
import vllm

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument(
        "-o", "--output", type=str, default=f"{ROOT}/schemas", help="Output directory"
    )
    args = args.parse_args()

    output_dir = pathlib.Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_schema = pydantic.TypeAdapter(qwen_vl_utils.VideoConfig).json_schema()
    (output_dir / "vision_config.json").write_text(json.dumps(vision_schema, indent=2))

    generation_schema = msgspec.json.schema(vllm.SamplingParams)
    (output_dir / "generation_config.json").write_bytes(
        msgspec.json.encode(generation_schema)
    )


if __name__ == "__main__":
    main()
