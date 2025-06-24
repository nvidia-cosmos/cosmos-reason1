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
from typing import Optional, Any, List, Dict
from torch.utils.data import Dataset, ConcatDataset, Subset
from datasets import load_dataset
from cosmos_rl.dispatcher.run_web_panel import main as launch_dispatcher
import cosmos_rl.utils.util as util
from cosmos_rl.policy.config import Config
from cosmos_rl.dispatcher.algo.reward import single_choice_reward_fn, format_reward_fn
from transformers import AutoTokenizer 
from cosmos_rl.utils.util import basename_from_modelpath
from cosmos_rl.dispatcher.data.packer.base import DataPacker
from cosmos_rl.dispatcher.data.packer.qwen2_5_vlm_data_packer import Qwen2_5_VLM_DataPacker
from cosmos_rl.utils.logging import logger
import toml
import argparse

FPS = 1
MAX_PIXELS = 81920


class CosmosGRPODataset(Dataset):
    def get_mm_files_paths(self, dataset_name: str, dataset_subset: str):
        cosmos_cache_dir = os.environ.get(
            "COSMOS_CACHE", os.path.join(os.path.expanduser("~"), ".cache/cosmos/")
        )
        video_clips_path = os.path.join(
            cosmos_cache_dir,
            "datasets",
            basename_from_modelpath(dataset_name),
            dataset_subset,
            "video_clips",
        )
        if not os.path.exists(video_clips_path):
            raise FileNotFoundError(
                f"Dataset directory {video_clips_path} does not exist. Please check the dataset path."
            )
        mm_files_paths = {}
        for root, _, files in os.walk(video_clips_path):
            for file in files:
                if file.endswith(
                    (".mp4", ".avi", ".mov")
                ):  # Common video extensions
                    mm_files_paths[file] = os.path.join(root, file)
        return mm_files_paths

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = load_dataset(config.train.train_policy.dataset.name, config.train.train_policy.dataset.subset)
        if config.train.train_policy.dataset.split:
            if isinstance(config.train.train_policy.dataset.split, list):
                dataset_list = []
                for split_name in config.train.train_policy.dataset.split:
                    dataset_list.append(self.dataset[split_name])
                self.dataset = ConcatDataset(dataset_list)
            else:
                assert isinstance(config.train.train_policy.dataset.split, str)
                self.dataset = self.dataset[config.train.train_policy.dataset.split]
        util.prepare_cosmos_data(dataset=config.train.train_policy.dataset, fps=FPS, max_pixels=MAX_PIXELS)
        self.mm_files_paths = self.get_mm_files_paths(config.train.train_policy.dataset.name, config.train.train_policy.dataset.subset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        '''
            Return a tuple of (prompt, reference answer)
        '''
        payload = self.dataset[idx]
        grpo_config = self.config.train.train_policy
        
        system_prompt = ""

        choices = payload["qa_pairs"]["index2ans"]
        user_prompt = (
            payload["qa_pairs"]["question"]
            + "\n"
            + "\n".join([f"({i}) {choice}" for i, choice in choices.items()])
        )
        user_prompt += (
            "\nAnswer with the option's letter from the given choices directly."
        )
        user_prompt += "\nPlease answer the question in the following format: <think> your reasoning </think> <answer> your answer </answer>."

        user_conv = [
            {
                "type": "text",
                "text": user_prompt,
            },
        ]

        if "video" in payload or "image" in payload:
            if "video" in payload:
                multi_modal_content = {
                    "type": "video",
                    "video": self.mm_files_paths[payload["video"].split("/")[-1]],
                    "max_pixels": MAX_PIXELS,
                    "fps": FPS,
                }
            else:
                multi_modal_content = {
                    "type": "image",
                    "image": self.mm_files_paths[payload["image"].split("/")[-1]],
                }
            user_conv.insert(0, multi_modal_content)

        conversations = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_conv,
            },
        ]

        return conversations

    def get_reference_answer(self, idx: int) -> str:
        return self.dataset[idx]["qa_pairs"]['answer']

class CosmosGRPOValDataset(CosmosGRPODataset):
    '''
    This is a validation dataset for Cosmos GRPO, which is used to evaluate the performance of the model.
    It should be used in the launcher to evaluate the model during training.
    '''
    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        if not config.train.enable_validation:
            logger.warning("Validation is not enabled in the config. Skipping setup for CosmosGRPOValDataset.")
            return

        self.config = config
        self.tokenizer = tokenizer
        self.dataset = load_dataset(config.validation.dataset.name, config.validation.dataset.subset)

        if config.validation.dataset.split:
            if isinstance(config.validation.dataset.split, list):
                dataset_list = []
                for split_name in config.validation.dataset.split:
                    dataset_list.append(self.dataset[split_name])
                self.dataset = ConcatDataset(dataset_list)
            else:
                assert isinstance(config.validation.dataset.split, str)
                self.dataset = self.dataset[config.validation.dataset.split]
        # Prepare the data for Cosmos GRPO
        # This is a hack to make the dataset compatible with the training data
        # Change the training dataset name and subset to utilize the same data preparation logic
        util.prepare_cosmos_data(dataset=config.validation.dataset, fps=FPS, max_pixels=MAX_PIXELS)
        self.mm_files_paths = self.get_mm_files_paths(config.train.train_policy.dataset.name, config.train.train_policy.dataset.subset)

def custom_reward_fn(to_be_evaluated: str, reference: Optional[str] = None, *args, **kwargs) -> float:
    return sum([single_choice_reward_fn(to_be_evaluated, reference, *args, **kwargs), format_reward_fn(to_be_evaluated, reference, *args, **kwargs)])

class DemoDataPacker(DataPacker):
    '''
    This is a demo data packer that wraps the underlying data packer of the selected model.
    This is meaningless for this example, but useful for explaining:
        - how dataset data is processed and collated into a mini-batch for rollout engine;
        - how rollout output is processed and collated into a mini-batch for policy model;
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check source code of Qwen2_5_VLM_DataPacker to see how it's implemented
        self.underlying_data_packer = Qwen2_5_VLM_DataPacker()

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        '''
        This method is optional and get called by launcher after being mounted
        `config`: config;
        `tokenizer`: tokenizer;
        '''
        super().setup(config, tokenizer, *args, **kwargs)
        self.underlying_data_packer.setup(config, tokenizer, *args, **kwargs)

    def get_rollout_input(self, item: Any) -> Any:
        '''
        Convert dataset item into what rollout engine (e.g. vllm) expects
        '''
        return self.underlying_data_packer.get_rollout_input(item)

    def rollout_collate_fn(self, items: List[Any]) -> Any:
        '''
        Collate the rollout inputs into a mini-batch for rollout engine
        '''
        return self.underlying_data_packer.rollout_collate_fn(items)

    def get_policy_input(self, item: Any, rollout_output: str, n_ignore_prefix_tokens: int = 0) -> Any:
        '''
        Process samples & rollout output before collating them into a mini-batch
        '''
        return self.underlying_data_packer.get_policy_input(item, rollout_output, n_ignore_prefix_tokens)

    def policy_compute_max_len(self, processed_samples: List[Any]) -> int:
        '''
        Compute the maximum sequence length of the mini-batch
        '''
        return self.underlying_data_packer.policy_compute_max_len(processed_samples)

    def policy_collate_fn(self, processed_samples: List[Any], computed_max_len: int) -> Dict[str, Any]:
        '''
        Collate the mini-batch into the kwargs required by the policy model
        '''
        return self.underlying_data_packer.policy_collate_fn(processed_samples, computed_max_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = toml.load(f)
    config = Config.from_dict(config)

    dataset = CosmosGRPODataset()
    val_dataset = CosmosGRPOValDataset() if config.train.enable_validation else None
    launch_dispatcher(
        dataset=dataset,
        reward_fns=[custom_reward_fn],
        data_packer=DemoDataPacker(),
        val_dataset=val_dataset,
        val_reward_fns=[custom_reward_fn],
        val_data_packer=DemoDataPacker(),
    )