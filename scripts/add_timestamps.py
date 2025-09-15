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

# https://docs.astral.sh/uv/guides/scripts/#using-a-shebang-to-create-an-executable-file
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cosmos-reason1-utils",
#   "imageio_ffmpeg>=0.6.0",
#   "torch>=2.7.1",
#   "torchcodec>=0.6.0",
# ]
# [tool.uv.sources]
# cosmos-reason1-utils = {path = "../cosmos_reason1_utils", editable = true}
# torch = [
#   { index = "pytorch-cu128"},
# ]
# torchvision = [
#   { index = "pytorch-cu128"},
# ]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///

"""Overlay timestamps at the bottom of a video.

Example:

```shell
./scripts/add_timestamps.py --video assets/sample.mp4 -o outputs/sample_timestamped.mp4
```
"""
# ruff: noqa: E402

from cosmos_reason1_utils.script import init_script

init_script()

import argparse
import os
import pathlib

import imageio_ffmpeg
import torch
import torchcodec

from cosmos_reason1_utils.vision import (
    overlay_text_on_tensor,
)

ROOT = pathlib.Path(__file__).parents[1].resolve()
SEPARATOR = "-" * 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Video path.", required=True)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for the timestamped video.",
        required=True,
    )
    args = parser.parse_args()

    video_path = args.video

    dir_name = os.path.dirname(args.output)
    os.makedirs(dir_name, exist_ok=True)

    print(SEPARATOR)
    print("Reading video to tensor.")
    video_tensor = _read_video(video_path)

    print(SEPARATOR)
    metadata = _get_metadata(video_path)
    print("Adding watermark to each frame.")
    video_tensor = overlay_text_on_tensor(video_tensor, fps=metadata["fps"])

    print(SEPARATOR)
    print(f"Saving frames to {args.output}.")

    pix_fmt = metadata["pix_fmt"]
    suffix = "(progressive)"
    if pix_fmt.endswith(suffix):
        pix_fmt = pix_fmt[: -len(suffix)]
    _write_video(
        video_tensor,
        args.output,
        fps=metadata["fps"],
        pix_fmt_out=pix_fmt,
        bitrate=str(metadata["bitrate"]),
    )


def _read_video(video_path):
    decoder = torchcodec.decoders.VideoDecoder(video_path)

    # Pre-allocate output tensor using metadata.
    metadata = decoder.metadata
    num_frames = metadata.num_frames
    height, width = metadata.height, metadata.width
    channels = 3

    # Preallocate tensor: [T, C, H, W].
    video_tensor = torch.empty((num_frames, channels, height, width), dtype=torch.uint8)

    for idx, frame in enumerate(decoder):
        assert frame.shape == (3, height, width)
        video_tensor[idx, ...] = frame

    return video_tensor


def _get_metadata(video_path):
    # Unfortunately, neither `VideoDecoder.metadata` nor `imageio_ffmpeg` return everything we want.
    reader = imageio_ffmpeg.read_frames(video_path)
    try:
        # The first call yields a metadata dict.
        merged_metadata = next(reader)
    finally:
        reader.close()

    metadata = torchcodec.decoders.VideoDecoder(video_path).metadata
    merged_metadata["bitrate"] = int(metadata.bit_rate)

    return merged_metadata


def _write_video(video_tensor, output_path, **ffmpeg_kwargs):
    # Expected channel order is HWC (input is TCHW).
    video_tensor = video_tensor.permute(0, 2, 3, 1).contiguous()
    # This takes the height added for the timestamps into account.
    height, width = video_tensor.shape[1], video_tensor.shape[2]
    generator = imageio_ffmpeg.write_frames(
        output_path,
        size=(width, height),
        **ffmpeg_kwargs,
    )
    try:
        # Seed the generator.
        generator.send(None)
        generator.send(video_tensor.cpu().numpy())
    finally:
        generator.close()


if __name__ == "__main__":
    main()
