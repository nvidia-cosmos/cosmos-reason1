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
from transformers import AutoTokenizer
from cosmos_reason1.dispatcher.data.packer.decoder_only_llm_data_packer import DecoderOnlyLLMDataPacker
from cosmos_reason1.dispatcher.data.packer.base import DataPacker
from cosmos_reason1.utils.modelscope import modelscope_load_dataset
from cosmos_reason1.utils.logging import logger


class MathDataset(Dataset):
    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        '''
        This method is optional and get called by launcher after being mounted
        `config`: config;
        `tokenizer`: tokenizer;
        '''
        self.config = config
        self.tokenizer = tokenizer
        modelscope_dataset_if_enabled = modelscope_load_dataset(config.train.train_policy.dataset.name, subset_name="default", split='train')
        if modelscope_dataset_if_enabled is None:
            self.dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
        else:
            self.dataset = modelscope_dataset_if_enabled


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        assert hasattr(self, "tokenizer"), "`self.tokenizer` should be set by the launcher"
        question = self.dataset[idx]["problem"]
        assert isinstance(
            question, str
        ), f"Prompt should be a string, but got {type(question)}, {question}"
        # Convert to templated prompt
        conversation = [
            {
                "role": "user",
                "content": question + " Let's think step by step and output the final answer within \\boxed{}.",
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
        return self.dataset[idx]["solution"]


class MathValDataset(MathDataset):
    '''
    This is a validation dataset for Math, which is used to evaluate the performance of the model.
    It should be used in the launcher to evaluate the model during training.
    '''
    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        if not config.train.enable_validation:
            logger.warning("Validation is not enabled in the config. Skipping setup for MathValDataset.")
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



# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def strip_string(string):

    def fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:  # noqa: E722
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:  # noqa: E722
            return string

    def remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def custom_reward_fn(to_be_evaluated: str, reference: Optional[Any] = None, *args, **kwargs) -> float:
    assert isinstance(reference, str), "Reference answer should be a string"
    reward = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(to_be_evaluated)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            reference_answer = remove_boxed(last_boxed_only_string(reference))
            if is_equiv(answer, reference_answer):
                reward = 1.0
    except Exception as e:
        logger.error(f"Error in custom reward function: {e}")
        logger.error(f"to_be_evaluated: {to_be_evaluated}")
        logger.error(f"reference: {reference}")
    return reward


class MathDataPacker(DataPacker):
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
    dataset = MathDataset()
    val_dataset = MathValDataset()
    launch_dispatcher(
        dataset=dataset,
        # Override the reward functions defined in toml
        reward_fns=[custom_reward_fn],
        # Optional: if not provided, the default data packer of the selected model will be used
        data_packer=MathDataPacker(),
        val_dataset=val_dataset,
        val_reward_fns=[custom_reward_fn],
        val_data_packer=MathDataPacker(),
    )