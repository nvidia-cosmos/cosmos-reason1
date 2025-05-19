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
import shutil
import torch
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor

from cosmos_reason1.utils.logging import logger


class DiskCache:
    """use disk cache datasets's preprocess data"""

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        os.makedirs(self.cache_path, exist_ok=True)

        self.max_concurrent_tasks = 4
        self.max_files_per_dir = 10000
        self.executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="disk_cache"
        )

    def __cache_ojb_path(self, idx: int) -> str:
        # avoid too many files in a single directory, limit by filesystem
        # TODO(zjx): we can reduce files count by merge data
        subdir = str(idx // self.max_files_per_dir)
        subdir_path = os.path.join(self.cache_path, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        return os.path.join(subdir_path, f"{idx}.pt")

    def __save_to_disk(self, path: str, obj: Any) -> None:
        def save_obj_helper(obj, path):
            tmp_file = f"{path}.tmp"
            try:
                # Save to temporary file first
                torch.save(obj, tmp_file)
                os.rename(tmp_file, path)
            except Exception:
                # Clean up temp file if it exists
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

        # Save to temporary file first
        self.executor.submit(save_obj_helper, obj, path)

    def set(self, idx: int, obj: Any) -> None:
        if self.executor._work_queue.qsize() > self.max_concurrent_tasks:
            return

        # Create a task to run the async operation
        cachePath = self.__cache_ojb_path(idx)
        self.__save_to_disk(cachePath, obj)

    def get(self, idx: int) -> Optional[Any]:
        cachePath = self.__cache_ojb_path(idx)
        if os.path.exists(cachePath):
            try:
                return torch.load(cachePath)
            except Exception as e:
                logger.error(f"Failed to load cache file {cachePath}: {e}")
                return None
        return None

    def clear(self) -> None:
        self.executor.shutdown(wait=False)

        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path)
