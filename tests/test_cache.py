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
import time
from cosmos_reason1.utils.cache import DiskCache


def test_disk_cache():
    cache_dir = "/tmp/test_cache"

    cache = DiskCache(cache_dir)
    cache.set(0, "test")
    # wait for the cache to be saved
    time.sleep(1)

    assert cache.get(0) == "test"

    # test clean
    cache.clear()
    assert not os.path.exists(cache_dir)


if __name__ == "__main__":
    test_disk_cache()
