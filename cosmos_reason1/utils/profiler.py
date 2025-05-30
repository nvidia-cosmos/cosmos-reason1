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


import concurrent.futures as futures
from typing import List
import os
import torch
from functools import partial
import requests

from cosmos_reason1.utils.logging import logger
from cosmos_reason1.policy.config import Config as CosmosConfig
from cosmos_reason1.utils.parallelism import ParallelDims
from cosmos_reason1.utils.network_util import make_request_with_retry
from cosmos_reason1.utils.checkpoint import upload_folder_to_s3
from cosmos_reason1.utils import constant


class CosmosProfiler:
    WAIT_STEPS = 1
    WARMUP_STEPS = 1

    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        replica_name: str,
        alternative_urls: List[str],
    ):
        self.config = config
        self.replica_name = replica_name
        self.profiler_config = config.profiler
        self.global_rank = parallel_dims.global_rank
        self.skip_prof = False
        self.saved = False
        self.is_started = False
        self.profiler = None
        self.thread_pool = None
        self.alternative_urls = alternative_urls
        # We do not want the timestamp part of output dir for profiling data.
        output_dir = os.path.dirname(config.train.output_dir)
        self.output_dir = os.path.join(
            output_dir, "profile_trace", f"{self.replica_name}_{self.global_rank}"
        )

        self._config_validate()
        if (
            self.profiler_config.enable_profiler
            and self.global_rank in self.profiler_config.rank_filter
        ):
            logger.info(f"[Profiler] init profiler for rank {self.global_rank}")
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=self.WAIT_STEPS,
                    warmup=self.WARMUP_STEPS,
                    active=self.profiler_config.active_steps,
                    repeat=1,
                ),
                record_shapes=True,
                with_stack=True,
            )
            self.thread_pool = futures.ThreadPoolExecutor(max_workers=4)

    def _config_validate(self):
        if self.profiler_config.enable_profiler:
            if len(self.profiler_config.rank_filter) == 0:
                logger.warning(
                    "[Profiler] rank_filter is empty, set rank_filter to [0]"
                )
                self.profiler_config.rank_filter = [0]
            if self.profiler_config.active_steps <= 0:
                raise ValueError(
                    "[Profiler] step that profiler traces must be positive."
                )

    def check(self):
        return self.profiler is not None and not self.skip_prof

    def start(self):
        if self.check():
            if not self.is_started:
                logger.info(f"[Profiler] start to trace for rank: {self.global_rank}")
                self.profiler.start()
                self.is_started = True

    def stop(self):
        if self.check() and self.is_started:
            self.profiler.stop()
            self.is_started = False

    def step(self):
        if self.check() and self.is_started:
            logger.debug(
                f"[Profiler] Step ahead for rank: {self.global_rank}, step_num: {self.profiler.step_num}"
            )
            self.profiler.step()
            if self.should_stop():
                self.stop_and_save()

    def get_total_steps_needed(self):
        if self.profiler is not None:
            warm_up_steps = self.WARMUP_STEPS
            wait_steps = self.WAIT_STEPS
            active_steps = self.profiler_config.active_steps
            num_steps = wait_steps + warm_up_steps + active_steps
            return num_steps
        return 0

    def should_stop(self):
        if self.check():
            return self.profiler.step_num >= self.get_total_steps_needed()
        return True

    def save(self):
        if self.profiler is not None and not self.saved:
            os.makedirs(self.output_dir, exist_ok=True)
            trace_file_name = "trace.json"
            trace_file_path = os.path.join(self.output_dir, trace_file_name)
            logger.info(
                f"[Profiler] save trace for rank: {self.global_rank} to file: {trace_file_path} after {self.profiler.step_num} steps."
            )
            # save the trace asynchronously
            self.profiler.export_chrome_trace(trace_file_path)

            # report to the controller
            self.report_to_controller(trace_file_path)

            self.saved = True

    def report_to_controller(self, trace_file_path: str):
        abs_path = os.path.abspath(trace_file_path)
        if self.config.train.ckpt.upload_s3:
            # FIXME: (lms) Now we do not have the authority to S3 storage, test it later.
            logger.info(
                f"Uploading the trace to s3: {self.config.train.ckpt.s3_bucket}..."
            )
            self.thread_pool.submit(
                upload_folder_to_s3,
                abs_path,
                self.config.train.ckpt.s3_bucket,
                self.config.train.ckpt.s3_prefix,
            )
        else:
            # just leave the trace file in local disk.
            pass
        post_func = partial(
            requests.post,
            json={
                "replica_name": self.replica_name,
                "trace_path": abs_path,
                "global_rank": self.global_rank,
            },
        )

        report_func = partial(
            make_request_with_retry,
            post_func,
            self.alternative_urls,
            max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
        )
        self.thread_pool.submit(report_func)

    def stop_and_save(self):
        if self.check():
            self.stop()
            self.save()
            # shutdown the thread pool
            self.thread_pool.shutdown(wait=True)
        self.stop_trace()

    def stop_trace(self):
        if self.check():
            logger.info(f"[Profiler] stop trace for rank: {self.global_rank}")
            self.skip_prof = True
            self.profiler = None
