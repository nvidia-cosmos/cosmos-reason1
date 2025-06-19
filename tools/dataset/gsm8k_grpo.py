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
from torch.utils.data import Dataset, ConcatDataset, Subset
from datasets import load_dataset
from cosmos_reason1.dispatcher.run_web_panel import main as launch_dispatcher
from cosmos_reason1.policy.config import Config
from cosmos_reason1.dispatcher.algo.reward import gsm8k_reward_fn
from transformers import AutoTokenizer
from cosmos_reason1.dispatcher.data.packer.decoder_only_llm_data_packer import DecoderOnlyLLMDataPacker
from cosmos_reason1.dispatcher.data.packer.base import DataPacker
from cosmos_reason1.utils.modelscope import modelscope_load_dataset
from cosmos_reason1.utils.logging import logger


class GSM8kDataset(Dataset):
    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        '''
        This method is optional and get called by launcher after being mounted
        `config`: config;
        `tokenizer`: tokenizer;
        '''
        self.config = config
        self.tokenizer = tokenizer
        modelscope_dataset_if_enabled = modelscope_load_dataset('AI-ModelScope/gsm8k', subset_name='main', split='train')
        if modelscope_dataset_if_enabled is None:
            self.dataset = load_dataset("openai/gsm8k", "main", split="train")
        else:
            self.dataset = modelscope_dataset_if_enabled


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
        question = self.dataset[idx]["question"]
        assert isinstance(
            question, str
        ), f"Prompt should be a string, but got {type(question)}, {question}"
        # Convert to templated prompt
        conversation = [
            {
                "role": "user",
                "content": f"{question} Let\'s think step by step and output the final answer after \"####\".",
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def get_reference_answer(self, idx: int) -> Any:
        '''
        This is mandatory for GRPO to get a reference answer for reward computation.
        '''
        return self.dataset[idx]["answer"]


class GSM8kValDataset(GSM8kDataset):
    '''
    This is a validation dataset for GSM8K, which is used to evaluate the performance of the model.
    It should be used in the launcher to evaluate the model during training.
    '''
    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        if not config.train.enable_validation:
            logger.warning("Validation is not enabled in the config. Skipping setup for GSM8kValDataset.")
            return

        self.config = config
        self.tokenizer = tokenizer
        if not config.validation.dataset.name:
            config.validation.dataset.name = config.train.train_policy.dataset.name
            config.validation.dataset.subset = config.train.train_policy.dataset.subset
            config.validation.dataset.revision = config.train.train_policy.dataset.revision
        if not config.validation.dataset.test_split:
            config.validation.dataset.test_split = config.train.train_policy.dataset.train_split
        if not config.validation.dataset.test_size:
            config.validation.dataset.test_size = config.train.train_policy.dataset.test_size

        self.dataset = load_dataset(config.validation.dataset.name, config.validation.dataset.subset)
        if config.validation.dataset.test_split:
            if isinstance(config.validation.dataset.test_split, list):
                dataset_list = []
                for split_name in config.validation.dataset.test_split:
                    dataset_list.append(self.dataset[split_name])
                self.dataset = ConcatDataset(dataset_list)
            else:
                assert isinstance(config.validation.dataset.test_split, str)
                self.dataset = self.dataset[config.validation.dataset.test_split]
        if config.validation.dataset.test_size is not None:
            if isinstance(config.validation.dataset.test_size, float):
                n_test_samples = int(
                    len(self.dataset) * config.validation.dataset.test_size
                )
            else:
                n_test_samples = config.validation.dataset.test_size
            n_test_samples = max(min(n_test_samples, len(self.dataset)), 1)

            # Prepare the validation dataset by taking the first `n_test_samples` samples
            indices = list(range(len(self.dataset)))
            val_indices = indices[:n_test_samples]
            self.dataset = Subset(self.dataset, val_indices)


def custom_reward_fn(to_be_evaluated: str, reference: Optional[Any] = None, *args, **kwargs) -> float:
    assert isinstance(reference, str), "Reference answer should be a string"
    reward = gsm8k_reward_fn(to_be_evaluated, reference, *args, **kwargs)
    # Add more reward functions here
    # ...
    return reward


class DemoDataPacker(DataPacker):
    '''
    This is a demo data packer that wraps the underlying data packer of the selected model.
    This is meaningless for this example, but useful for explaining:
        - how dataset data is processed and collated into a mini-batch for rollout engine;
        - how rollout output is processed and collated into a mini-batch for policy model;
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check source code of DecoderOnlyLLMDataPacker to see how it's implemented
        self.underlying_data_packer = DecoderOnlyLLMDataPacker()

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
    dataset = GSM8kDataset()
    val_dataset = GSM8kValDataset()
    launch_dispatcher(
        dataset=dataset,
        # Override the reward functions defined in toml
        reward_fns=[custom_reward_fn],
        # Optional: if not provided, the default data packer of the selected model will be used
        data_packer=DemoDataPacker(),
        val_dataset=val_dataset,
        val_reward_fns=[custom_reward_fn],
        val_data_packer=DemoDataPacker(),
    )