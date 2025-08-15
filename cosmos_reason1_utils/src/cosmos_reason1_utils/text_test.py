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

from cosmos_reason1_utils.text import (
    create_conversation,
    extract_tagged_text,
    set_vision_kwargs,
)


def test_create_conversation():
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the capital of France?"
    images = ["image1.png", "image2.png"]
    videos = ["video1.mp4", "video2.mp4"]
    vision_kwargs = {"max_pixels": 10}
    conversation = create_conversation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        videos=videos,
    )
    set_vision_kwargs(conversation, vision_kwargs)
    assert conversation == [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]} | vision_kwargs,
                {"type": "image", "image": images[1]} | vision_kwargs,
                {"type": "video", "video": videos[0]} | vision_kwargs,
                {"type": "video", "video": videos[1]} | vision_kwargs,
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def test_extract_tagged_text():
    text = """Intro text
<question>
What is the capital of France?
</question>
Middle text
<answer>
Paris
</answer>
End text
"""
    result, remaining = extract_tagged_text(text)
    assert result == {
        "question": ["\nWhat is the capital of France?\n"],
        "answer": ["\nParis\n"],
    }
    assert remaining == ["Intro text\n", "\nMiddle text\n", "\nEnd text\n"]
