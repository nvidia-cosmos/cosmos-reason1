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

import collections
from typing import Any
import pydantic
from pydantic import Field
import re

"""Text processing utilities."""


class PromptConfig(pydantic.BaseModel):
    """Prompt config."""

    model_config = pydantic.ConfigDict(extra="forbid")

    system_prompt: str = Field(default="", description="System prompt")
    user_prompt: str = Field(default="", description="User prompt")


def create_conversation(
    *,
    system_prompt: str = "",
    user_prompt: str = "",
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
    vision_kwargs: dict,
) -> list[dict]:
    """Create chat conversation.

    Args:
        system_prompt: System prompt.
        user_prompt: User prompt.
        images: List of images.
        videos: List of videos.
        vision_kwargs: Keyword arguments for vision processor (see `cosmos_reason1_utils.vision.VisionConfig`).

    Returns:
        conversation: Chat conversation.
    """
    user_content = []
    if images is not None:
        for image in images:
            user_content.append({"type": "image", "image": image} | vision_kwargs)
    if videos is not None:
        for video in videos:
            user_content.append({"type": "video", "video": video} | vision_kwargs)
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": user_content})
    return conversation


def extract_structured_text(text: str) -> tuple[dict[str, list[str]], list[str]]:
    """Extract structured text between <key> and </key> tags.

    Ignores unclosed tags and tries to extract as much text as possible.

    Example:

    ```python
    text = '''Intro text
    <question>
    What is the capital of France?
    </question>
    Middle text
    <answer>
    Paris
    </answer>
    End text
    '''
    result, remaining = extract_structured_text(text)
    assert result == {
        "question": ["\nWhat is the capital of France?\n"],
        "answer": ["\nParis\n"]
    }
    assert remaining == ["Intro text\n", "\nMiddle text\n", "\nEnd text\n"]
    ```

    Args:
        text: Text to extract from.

    Returns:
        result: Mapping from key to list of extracted texts.
        remaining: Remaining texts.
    """
    open_tag_pattern = re.compile(r"<([a-zA-Z]*?)>")

    result: dict[str, list[str]] = collections.defaultdict(list)
    remaining: list[str] = []
    start = 0
    while start < len(text):
        # Find next open tag
        match = open_tag_pattern.search(text, start)
        if match is None:
            remaining.append(text[start:])
            break
        remaining.append(text[start:match.start()])
        start = match.end()
        key = match.group(1)

        # Find corresponding close tag
        close_tag = f"</{key}>"
        end = text.find(f"</{key}>", start)
        if end == -1:
            # Ignore unclosed tags
            continue
        result[key].append(text[start:end])
        start = end + len(close_tag)
    return dict(result), remaining
