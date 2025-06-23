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


from typing import Optional, Any, List, Dict
from torch.utils.data import Dataset
from datasets import load_dataset
from cosmos_reason1.dispatcher.run_web_panel import main as launch_dispatcher
from cosmos_reason1.policy.config import Config
from cosmos_reason1.dispatcher.algo.reward import direct_math_reward_fn, overlong_reward_fn
from transformers import AutoTokenizer
from torch.utils.data import ConcatDataset

class MathDapoDataset(Dataset):
    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        '''
        This method is optional and get called by launcher after being mounted
        `config`: config;
        `tokenizer`: tokenizer;
        '''
        self.config = config
        self.tokenizer = tokenizer

        # This demo is only for DAPO-Math-17k dataset
        assert "DAPO-Math-17k" in config.train.train_policy.dataset.name
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        '''
        For DecoderOnlyLLMDataPacker, it should either return:
        - raw text prompt to be converted into input_ids by both rollout and policy models;
        - conversation format:
        ```
        [
            {
                "role": "system",
                "content": "You are an AI math expert, you will be given a question and required to answer. "
            }
            ...
        ]
        ```
        '''
        assert hasattr(self, "tokenizer"), "`self.tokenizer` should be set by the launcher"
        conversation = self.dataset[idx]["prompt"]
        assert isinstance(
            conversation, list
        ), f"Prompt should be a string, but got {type(conversation)}， {conversation}"
        
        # return the conversation to be converted into input_ids by the data packer
        return conversation

    def get_reference_answer(self, idx: int) -> Any:
        '''
        This is mandatory for GRPO to get a reference answer for reward computation.
        '''
        return self.dataset[idx]["reward_model"]["ground_truth"]

def custom_reward_fn(to_be_evaluated: str, reference: Optional[Any] = None, *args, **kwargs) -> float:
    assert isinstance(reference, str), "Reference answer should be a string"
    reward = sum([direct_math_reward_fn(to_be_evaluated, reference, *args, **kwargs), overlong_reward_fn(to_be_evaluated, reference, *args, **kwargs)])
    return reward

if __name__ == "__main__":
    launch_dispatcher(
        dataset=MathDapoDataset(),
        # Override the reward functions defined in toml
        reward_fns=[custom_reward_fn],
    )