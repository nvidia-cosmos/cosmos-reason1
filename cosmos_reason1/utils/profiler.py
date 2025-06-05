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
        self.enable_profile = config.profiler.enable_profiler
        self.global_rank = parallel_dims.global_rank
        self.initialized = False
        self.is_started = False
        self.is_finished = False
        self.profiled_times = 0
        self.profiler = None
        self.thread_pool = None
        self.alternative_urls = alternative_urls
        # We do not want the timestamp part of output dir for profiling data.
        output_dir = os.path.dirname(config.train.output_dir)
        self.output_dir = os.path.join(
            output_dir, "profile_trace", f"{self.replica_name}_{self.global_rank}"
        )
        self.thread_pool = None

        # static profiler config
        self.active_steps = config.profiler.sub_profiler_config.active_steps
        if len(config.profiler.sub_profiler_config.rank_filter) == 0:
            self.rank_filter = [0]
        else:
            self.rank_filter = config.profiler.sub_profiler_config.rank_filter
        self.record_shape = config.profiler.sub_profiler_config.record_shape
        self.profile_memory = config.profiler.sub_profiler_config.profile_memory
        self.with_stack = config.profiler.sub_profiler_config.with_stack
        self.with_modules = config.profiler.sub_profiler_config.with_modules

    def check(self):
        return self.profiler is not None and self.enable_profile

    def _validate_config(self):
        if len(self.rank_filter) == 0:
            self.rank_filter = [
                0,
            ]
        if self.active_steps <= 0:
            raise ValueError(
                f"[Profiler] active_steps must be positive, got {self.active_steps}"
            )

    def check_finished(self):
        finished = self.is_finished
        if self.is_finished:
            self.is_finished = False
        return finished

    def start(self):
        # start the profiler with static config
        if self.enable_profile:
            if self.global_rank not in self.rank_filter:
                return
            self._validate_config()
            if not self.initialized:
                self.initialized = True
                logger.info(f"[Profiler] init profiler for rank {self.global_rank}")
                self.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=self.WAIT_STEPS,
                        warmup=self.WARMUP_STEPS,
                        active=self.active_steps,
                        repeat=1,
                    ),
                    record_shapes=self.record_shape,
                    with_stack=self.with_stack,
                    with_modules=self.with_modules,
                    profile_memory=self.profile_memory,
                )
                self.thread_pool = futures.ThreadPoolExecutor(max_workers=4)

        if self.check():
            if not self.is_started:
                logger.info(f"[Profiler] start to trace for rank: {self.global_rank}")
                self.profiler.start()
                self.is_started = True
                self.is_finished = False

    def start_dynamic(
        self,
        active_steps: int,
        rank_filter: List[int],
        record_shape: bool,
        profile_memory: bool,
        with_stack: bool,
        with_modules: bool,
    ):
        if self.enable_profile:
            if self.global_rank not in rank_filter:
                return
            self._validate_config()
            if not self.initialized:
                self.initialized = True
                logger.info(f"[Profiler] init profiler for rank {self.global_rank}")
                self.active_steps = active_steps
                self.rank_filter = rank_filter
                self.record_shape = record_shape
                self.profile_memory = profile_memory
                self.with_stack = with_stack
                self.with_modules = with_modules
                self.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=self.WAIT_STEPS,
                        warmup=self.WARMUP_STEPS,
                        active=self.active_steps,
                        repeat=1,
                    ),
                    record_shapes=self.record_shape,
                    with_stack=self.with_stack,
                    with_modules=self.with_modules,
                    profile_memory=self.profile_memory,
                )
                self.thread_pool = futures.ThreadPoolExecutor(max_workers=4)

        if self.check():
            if not self.is_started:
                logger.info(f"[Profiler] start to trace for rank: {self.global_rank}")
                self.profiler.start()
                self.is_started = True
                self.is_finished = False

    def stop(self):
        if self.check():
            if self.is_started:
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
            active_steps = self.active_steps
            num_steps = wait_steps + warm_up_steps + active_steps
            return num_steps
        return 0

    def should_stop(self):
        if self.check():
            self.is_finished = self.profiler.step_num >= self.get_total_steps_needed()
            return self.profiler.step_num >= self.get_total_steps_needed()
        return True

    def stop_and_save(self):
        if self.check():
            self.stop()
            self.save()
            # shutdown the thread pool
            self.thread_pool.shutdown(wait=True)
            # relase the profiler
            del self.profiler
            self.profiler = None
            del self.thread_pool
            self.thread_pool = None
            self.is_started = False
            self.profiled_times += 1
            self.initialized = False
            logger.info(f"[Profiler] stop trace for rank: {self.global_rank}")

    def save(self):
        if self.check():
            os.makedirs(self.output_dir, exist_ok=True)
            trace_file_name = "trace.json.gz"
            trace_file_path = os.path.join(
                self.output_dir, f"{self.profiled_times}_{trace_file_name}"
            )
            logger.info(
                f"[Profiler] save trace for rank: {self.global_rank} to file: {trace_file_path} after {self.profiler.step_num} steps."
            )
            # save the trace asynchronously
            self.profiler.export_chrome_trace(trace_file_path)

            # report to the controller
            # only report the dir
            self.report_to_controller(os.path.dirname(trace_file_path))

    def report_to_controller(self, trace_file_dir: str):
        abs_path = os.path.abspath(trace_file_dir)
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
