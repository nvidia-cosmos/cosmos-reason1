from cosmos_reason1.dispatcher.data.packer.base import DataPacker
from typing import List, Any, Dict, Union
import torch


class DecoderOnlyLLMDataPacker(DataPacker):
    """
    Data protocol & processing logic for the decoder only LLM for SFT and RL training.
    """

    ConversationType = List[Dict[str, str]]

    class RLPolicyInput:
        input_ids: List[int]
        logprob_masks: List[int]

        def __init__(self, input_ids: List[int], logprob_masks: List[int]):
            self.input_ids = input_ids
            self.logprob_masks = logprob_masks

    def get_rollout_input(self, sample: Union[str, ConversationType]) -> str:
        """
        This is the default implementation for decoder only LLM data packer.
        It assumes that each sample is either a raw text or a conversation format list.
        """
        # 1. if item is a string, then assume it is a raw text
        if isinstance(sample, str):
            return sample
        # 2. if item is a list, then assume it is in conversation format of:
        # [
        #     {
        #         "role": "user",
        #         "content": "..."
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "..."
        #     }
        # ]
        else:
            assert isinstance(sample, list), "All items should be list"
            # Check `role` and `content` in each item
            for x in sample:
                assert isinstance(x, dict), "Each item should be a dict"
                assert "role" in x, "Each item should have 'role'"
                assert "content" in x, "Each item should have 'content'"

            # Apply template to each item
            prompt = self.tokenizer.apply_chat_template(
                sample,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt

    def get_policy_input(
        self,
        sample: Union[str, ConversationType],
        completion: str,
    ) -> RLPolicyInput:
        """
        Default text policy input packer.
        Only support raw text input.
        """
        assert isinstance(completion, str), "Completion should be a string"

        # Reuse the same logic as get_rollout_input to get raw text prompts
        prompt = self.get_rollout_input(sample)
        assert isinstance(prompt, str), "Prompt should be a string"

        input_ids = self.tokenizer(prompt).input_ids  # Don't pad yet

        completion_ids = self.tokenizer(completion).input_ids

        return DecoderOnlyLLMDataPacker.RLPolicyInput(
            input_ids=input_ids + completion_ids,
            logprob_masks=[0] * (len(input_ids) - 1) + [1] * len(completion_ids) + [0],
        )

    def policy_compute_max_len(self, processed_samples: List[RLPolicyInput]) -> int:
        return max([len(x.input_ids) for x in processed_samples])

    def policy_collate_fn(
        self, processed_samples: List[RLPolicyInput], computed_max_len: int
    ) -> Dict[str, Any]:
        input_ids = [x.input_ids for x in processed_samples]
        logprob_masks = [x.logprob_masks for x in processed_samples]
        assert len(input_ids) == len(
            logprob_masks
        ), "The length of input_ids, and logprob_masks should be the same"
        device = torch.cuda.current_device()

        collated_dict = {}
        collated_dict["input_ids"] = torch.tensor(
            [
                x + [self.tokenizer.pad_token_id] * (max(0, computed_max_len - len(x)))
                for x in input_ids
            ],
            dtype=torch.long,
        ).to(device)
        collated_dict["logprob_masks"] = torch.tensor(
            [x + [0] * (max(0, computed_max_len - len(x))) for x in logprob_masks],
            dtype=torch.bool,
        ).to(device)

        return collated_dict

    def sft_process_sample(self, sample: Union[str, List[Dict[str, str]]]) -> List[int]:
        """
        Process the sample into the format required by the SFT model.
        Accepts either raw text or conversation format.
        """
        # 1. if item is a string, then assume it is a raw text
        if isinstance(sample, str):
            token_ids = self.tokenizer(sample).input_ids
        # 2. if item is a list, then assume it is in conversation format of:
        # [
        #     {
        #         "role": "user",
        #         "content": "..."
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "..."
        #     }
        # ]
        else:
            assert isinstance(sample, list), "All items should be list, got: {}".format(
                sample
            )
            # Check `role` and `content` in each item
            for x in sample:
                assert isinstance(x, dict), "Each item should be a dict"
                assert "role" in x, "Each item should have 'role'"
                assert "content" in x, "Each item should have 'content'"

            # Apply template to each item
            token_ids = self.tokenizer.apply_chat_template(
                sample,
                return_assistant_tokens_mask=True,
                return_dict=True,
                add_generation_prompt=False,
            )["input_ids"]
        assert isinstance(token_ids, list), "token_ids should be a list"
        assert isinstance(token_ids[0], int), "Each item in token_ids should be an int"
        return token_ids

    def sft_compute_max_len(self, processed_samples: List[List[int]]) -> int:
        """
        Compute the maximum sequence length of the processed samples
        """
        max_len = max([len(x) for x in processed_samples])
        return max_len

    def sft_collate_fn(
        self,
        list_of_sample_tokens: List[List[int]],
        computed_max_len: int,
        pad_token_id: int,
        ignore_label_id: int,
    ) -> Dict[str, Any]:
        """
        Collate the processed samples into a minibatch dictionary passed to the SFT model.
        """
        # First truncate the samples to the computed_max_len
        list_of_sample_tokens = [x[:computed_max_len] for x in list_of_sample_tokens]
        # Then pad the samples to the computed_max_len
        input_ids = torch.tensor(
            [
                x + [pad_token_id] * (max(0, computed_max_len - len(x)))
                for x in list_of_sample_tokens
            ],
            dtype=torch.long,
        )
        # Model accept unshifted label_ids for loss computation
        label_ids = torch.tensor(
            [
                x + [ignore_label_id] * (max(0, computed_max_len - len(x)))
                for x in list_of_sample_tokens
            ],
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
        }
