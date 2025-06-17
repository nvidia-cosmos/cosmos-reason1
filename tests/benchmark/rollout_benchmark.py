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

import toml
import os
import time
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
import random

import torch.distributed as dist
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List, Tuple


from cosmos_reason1.rollout.vllm_rollout.vllm_rollout import vLLMRollout
from cosmos_reason1.utils.parallelism import ParallelDims
from cosmos_reason1.utils.distributed import init_distributed
from cosmos_reason1.policy.config import Config as CosmosConfig
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.dispatcher.data import CosmosDataset
from cosmos_reason1.utils import util


def main(args: argparse.Namespace):
    init_distributed()

    config = [
        None,
    ]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    config_file = args.config
    iteration_number = args.number

    if global_rank == 0:
        try:
            logger.info(f"Attempting to load configuration from {config_file}")
            with open(config_file, "r") as f:
                config_dict = toml.load(f)

            config[0] = CosmosConfig.from_dict(config_dict)
            logger.info(f"Successfully loaded configuration from {config_file}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load or parse config file {config_file}: {e}.",
                exc_info=True,
            )
    if world_size > 1:
        dist.broadcast_object_list(config, src=0, device=torch.device("cpu"))

    config = config[0]

    task_type = config.train.train_policy.type
    assert "grpo" in task_type.lower(), "Only GRPO needs rollout."

    tokenizer = AutoTokenizer.from_pretrained(config.policy.model_name_or_path)

    seed = config.rollout.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parallel_dims = ParallelDims.from_config(
        parallesim_config=config.rollout.parallelism
    )
    parallel_dims.build_mesh(device_type="cuda")

    batch_size = config.rollout.batch_size

    total_step = [
        0,
    ]
    if parallel_dims.global_rank == 0:
        # only the rank 0 is responsible for loading the dataset.
        try:
            # Load GRPO dataset
            # preprocess the dataset if is cosmos. This is highly consistent with the controller.
            is_cosmos = "cosmos" in config.train.train_policy.dataset.name.lower()
            if is_cosmos:
                util.prepare_cosmos_data(config=config)
            dataset = CosmosDataset(config=config)
            train_dataloader = DataLoader(dataset.train_set, batch_size=1, shuffle=True)
            logger.info(
                f"len(dataset.train_set): {len(dataset.train_set)}, batch_size: {batch_size}"
            )
            total_step[0] = (len(dataset.train_set) + batch_size - 1) // batch_size
            train_dataloader_iter = iter(train_dataloader)
        except Exception as e:
            logger.error(f"Fail to load dataset: {e}")

    if world_size > 1:
        dist.broadcast_object_list(total_step, src=0, device=torch.device("cpu"))

    total_step = total_step[0]

    logger.info(f"total_step: {total_step}, iteration_number: {iteration_number}")

    executed_step = min(iteration_number, total_step)
    if executed_step < 0:
        executed_step = total_step

    pbar = tqdm(total=executed_step, desc="[Benchmark] Step")

    rollout_engine = vLLMRollout(
        config,
        tokenizer=tokenizer,
        seed=seed,
        load_format="auto",
    )

    all_processed_tokens = 0
    time_of_all_iterations = 0

    for i in range(executed_step):
        temp = [None]
        if parallel_dims.global_rank == 0:
            pbar.update(1)
            prompt_id_and_payload_list: List[Tuple[int, str]] = []
            # get the prompt
            for _ in range(batch_size):
                try:
                    idx, prompt = next(train_dataloader_iter)
                except StopIteration:
                    logger.info("Rollout iterate over the dataset.")
                    break

                idx = idx.item() if isinstance(idx, torch.Tensor) else idx
                prompt_id_and_payload_list.append(
                    (
                        idx,
                        prompt[0]
                        if isinstance(prompt, list) or isinstance(prompt, tuple)
                        else prompt,
                    )
                )
            temp[0] = prompt_id_and_payload_list

        if world_size > 1:
            dist.broadcast_object_list(temp, src=0, device=torch.device("cpu"))

        prompt_id_and_payload_list = temp[0]

        # calculate the token number
        token_input_nums = []
        for _, prompt in prompt_id_and_payload_list:
            token_input_nums.append(len(tokenizer.encode(prompt)))

        # E2E rollout generation perf: prompt preprocessing + rollout generation
        start = time.perf_counter()
        completions_per_prompt = rollout_engine.rollout_generation(
            prompt_id_and_payload_list, stream=None
        )
        end = time.perf_counter()

        # calculate the output token number
        output_token_nums = []
        for completions in completions_per_prompt:
            token_num = 0
            for completion in completions:
                token_num += len(tokenizer.encode(completion))
            output_token_nums.append(token_num)

        total_output_tokens_num = sum(output_token_nums)
        total_tokens_num = total_output_tokens_num + sum(token_input_nums)

        all_processed_tokens += total_tokens_num
        time_of_all_iterations += end - start

        if (
            parallel_dims.tp_coord[0] == 0
            and parallel_dims.pp_coord[0] == parallel_dims.pp_coord[1] - 1
        ):
            elapsed_time = end - start
            logger.info(
                f"\nPerformance with batch size {batch_size}:\n"
                f"Total tokens/s: {total_tokens_num / elapsed_time:.2f}\n"
                f"Output tokens/s: {total_output_tokens_num / elapsed_time:.2f}"
            )
    if (
        parallel_dims.tp_coord[0] == 0
        and parallel_dims.pp_coord[0] == parallel_dims.pp_coord[1] - 1
    ):
        logger.info(
            f"All processed tokens: {all_processed_tokens} in {executed_step} iterations, time of all iterations: {time_of_all_iterations}, token/s: {all_processed_tokens / time_of_all_iterations:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="File path of the cosmos config.", required=True
    )
    parser.add_argument(
        "--number", type=int, help="Number of iterations.", required=True
    )

    args = parser.parse_args()

    main(args)
