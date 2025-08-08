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

import functools
import os
from pathlib import Path

import matplotlib.font_manager as fm
import numpy as np
import pydantic
import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field

"""Vision processing utilities."""


class VisionConfig(pydantic.BaseModel):
    """Config for vision processing.

    Source:
    https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py

    Attributes are sorted by priority. Higher priority attributes override lower
    priority attributes.
    """

    resized_height: int | None = Field(
        default=None, description="Max height of the image/video"
    )
    resized_width: int | None = Field(
        default=None, description="Max width of the image/video"
    )

    min_pixels: int | None = Field(
        default=None, description="Min frame pixels of the image/video"
    )
    max_pixels: int | None = Field(
        default=None, description="Max frame pixels of the image/video"
    )
    total_pixels: int | None = Field(
        default=None, description="Max total pixels of the image/video"
    )

    video_start: float | None = Field(
        None, description="Start time of the video (seconds)"
    )
    video_end: float | None = Field(None, description="End time of the video (seconds)")

    nframes: int | None = Field(
        default=None, description="Number of frames of the video"
    )

    fps: float | None = Field(default=None, description="FPS of the video")
    min_frames: int | None = Field(default=None, description="Min frames of the video")
    max_frames: int | None = Field(default=None, description="Max frames of the video")


def _tensor_to_pil_images(tensor: torch.Tensor) -> list[Image.Image]:
    """Convert a tensor to a list of PIL images.

    Args:
        tensor: Tensor with shape (C, H, W), (C, T, H, W) or (T, C, H, W)

    Returns:
        List of PIL images
    """
    # Check tensor shape and convert if needed
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.shape[0] == 3:  # (C, T, H, W)
        if tensor.shape[1] == 3:
            raise ValueError(f"Ambiguous shape: {tensor.shape}")
        # Convert to (T, C, H, W)
        tensor = tensor.permute(1, 0, 2, 3)

    # Convert to numpy array with shape (T, H, W, C)
    frames = tensor.permute(0, 2, 3, 1).cpu().numpy()

    # Ensure values are in the right range for PIL (0-255, uint8)
    if np.issubdtype(frames.dtype, np.floating):
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)

    return [Image.fromarray(frame) for frame in frames]


def _pil_images_to_tensor(images: list[Image.Image]) -> torch.Tensor:
    """Convert a list of PIL images to a tensor.

    Args:
        images: List of PIL images

    Returns:
        Tensor with shape (C, H, W) or (T, C, H, W)
    """
    tensor = torch.stack(
        [torchvision.transforms.functional.pil_to_tensor(image) for image in images],
        dim=0,
    )
    tensor.squeeze_(0)
    return tensor


def save_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a tensor as images to a directory.

    Args:
        tensor: Tensor with shape (C, H, W) or (T, C, H, W)
        path: Directory to save the images
    """
    os.makedirs(path, exist_ok=True)
    images = _tensor_to_pil_images(tensor)
    for i, image in enumerate(images):
        image.save(f"{path}/{i}.png")


class OverlayConfig(pydantic.BaseModel):
    """Config for overlaying text on images."""

    border_height: int = Field(
        default=28, description="Height of the black border in pixels."
    )
    temporal_path_size: int = Field(
        default=2, description="Number of positions to cycle through."
    )

    # Use 'DejaVu Sans Mono' font for better readability
    font_family: str = Field(
        default="DejaVu Sans Mono", description="Font family for the text."
    )
    font_size: int = Field(
        default=20, description="Font size for the text (in pixels)."
    )
    font_color: str = Field(default="white", description="Color of the text.")


@functools.cache
def _get_overlay_font_path(family: str) -> str:
    """Return the path to the font for overlaying text on images."""
    return fm.findfont(fm.FontProperties(family=family))


def overlay_text(
    images: list[Image.Image],
    *,
    fps: float | None = None,
    config: OverlayConfig = OverlayConfig(),  # noqa: B008
) -> list[Image.Image]:
    """Overlay text on a list of PIL images with black border.

    The timestamp position cycles through available positions.

    Args:
        images: List of PIL images to process
        fps: Frames per second
        config: Config for overlaying text

    Returns:
        List of PIL images with text overlay
    """
    font = ImageFont.truetype(
        _get_overlay_font_path(config.font_family), config.font_size
    )

    # Process each image
    processed_images = []

    for i, image in enumerate(images):
        # Get original dimensions
        width, height = image.size

        # Create new image with black border at the bottom
        new_height = height + config.border_height
        new_image = Image.new("RGB", (width, new_height), color="black")

        # Paste original image at the top
        new_image.paste(image, (0, 0))

        # Draw text on the black border
        draw = ImageDraw.Draw(new_image)

        # Calculate timestamp for current frame
        total_seconds = i / fps
        text = f"{total_seconds:.2f}s"

        # Get text dimensions
        try:
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(text, font=font)

        # Define available positions (cycling through horizontal positions)
        position_idx = i % config.temporal_path_size
        section_width = width // config.temporal_path_size

        # Calculate x position based on cycling position
        section_center_x = position_idx * section_width + section_width // 2
        text_x = section_center_x - text_width // 2

        # Ensure text doesn't go outside bounds
        text_x = max(0, min(text_x, width - text_width))

        # Center vertically in the border
        text_y = height + (config.border_height - text_height) // 2

        # Draw the single timestamp
        draw.text((text_x, text_y), text, fill=config.font_color, font=font)

        processed_images.append(new_image)

    return processed_images


def overlay_text_on_tensor(
    tensor: torch.Tensor,
    fps: float,
    config: OverlayConfig = OverlayConfig(),  # noqa: B008
) -> torch.Tensor:
    """Overlay text on a tensor.

    Args:
        tensor: Tensor with shape (C, H, W) or (T, C, H, W)
        fps: Frames per second
        config: Config for overlaying text
    Returns:
        Tensor with shape (C, H, W) or (T, C, H, W)
    """
    return _pil_images_to_tensor(
        overlay_text(_tensor_to_pil_images(tensor), fps=fps, config=config)
    )
