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

import matplotlib.font_manager as fm
import numpy as np
import pydantic
import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field

"""Vision processing utilities."""


class ImageConfig(pydantic.BaseModel):
    """Config for image processing."""

    min_pixels: int | None = Field(default=None, description="Min pixels of the image")
    max_pixels: int | None = Field(default=None, description="Max pixels of the image")

    resized_height: int | None = Field(
        default=None, description="Max height of the image"
    )
    resized_width: int | None = Field(
        default=None, description="Max width of the image"
    )


class VisionConfig(ImageConfig):
    """Config for image/video processing."""

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

    total_pixels: int | None = Field(
        default=None, description="Max pixels of the video"
    )


def _tensor_to_pil_images(video_tensor: torch.Tensor) -> list[Image.Image]:
    """Convert a video tensor to a list of PIL images.

    Args:
        video_tensor: Tensor with shape (C, T, H, W) or (T, C, H, W)

    Returns:
        List of PIL images
    """
    # Check tensor shape and convert if needed
    if video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:  # (C, T, H, W)
        # Convert to (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)

    # Convert to numpy array with shape (T, H, W, C)
    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()

    # Ensure values are in the right range for PIL (0-255, uint8)
    if video_np.dtype == np.float32 or video_np.dtype == np.float64:
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

    # Convert each frame to a PIL image
    pil_images = [Image.fromarray(frame) for frame in video_np]

    return pil_images


def _pil_images_to_tensor(images: list[Image.Image]) -> torch.Tensor:
    """Convert a list of PIL images to a video tensor.

    Args:
        pil_images: List of PIL images

    Returns:
        Tensor with shape (T, C, H, W)
    """
    return torch.stack(
        [torchvision.transforms.functional.pil_to_tensor(image) for image in images],
        dim=0,
    )


@functools.cache
def _get_overlay_font_path() -> str:
    """Return the path to the font for overlaying text on images."""
    # Use DejaVu Sans Mono font for better readability
    return fm.findfont(fm.FontProperties(family="DejaVu Sans Mono", style="normal"))


def overlay_text(
    images: list[Image.Image],
    *,
    fps: float,
    border_height: int = 28,  # this is due to patch size of 28
    temporal_path_size: int = 2,  # Number of positions to cycle through
    font_size: int = 20,
    font_color: str = "white",
) -> tuple[list[Image.Image], list[float]]:
    """Overlay text on a list of PIL images with black border.

    The timestamp position cycles through available positions.

    Args:
        images: List of PIL images to process
        fps: Frames per second
        border_height: Height of the black border in pixels (default: 28)
        temporal_path_size: Number of positions to cycle through (default: 2)
        font_size: Font size for the text (default: 20)
        font_color: Color of the text (default: "white")

    Returns:
        List of PIL images with text overlay
        List of timestamps
    """

    font = ImageFont.truetype(_get_overlay_font_path(), font_size)

    # Process each image
    processed_images = []

    for i, image in enumerate(images):
        # Get original dimensions
        width, height = image.size

        # Create new image with black border at the bottom
        new_height = height + border_height
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
        position_idx = i % temporal_path_size
        section_width = width // temporal_path_size

        # Calculate x position based on cycling position
        section_center_x = position_idx * section_width + section_width // 2
        text_x = section_center_x - text_width // 2

        # Ensure text doesn't go outside bounds
        text_x = max(0, min(text_x, width - text_width))

        # Center vertically in the border
        text_y = height + (border_height - text_height) // 2

        # Draw the single timestamp
        draw.text((text_x, text_y), text, fill=font_color, font=font)

        processed_images.append(new_image)

    return processed_images, [i / fps for i in range(len(images))]


def overlay_text_on_video(
    video_tensor: torch.Tensor, *, fps: float, **kwargs
) -> torch.Tensor:
    """Overlay text on a video tensor.

    Args:
        video_tensor: Tensor with shape (C, T, H, W) or (T, C, H, W)
        fps: Frames per second

    Returns:
        Tensor with shape (T, C, H, W)
    """
    return _pil_images_to_tensor(
        overlay_text(_tensor_to_pil_images(video_tensor), fps=fps, **kwargs)[0]
    )
