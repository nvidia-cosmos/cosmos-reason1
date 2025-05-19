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

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Callable, Union
import torch
from functools import cached_property
from cosmos_reason1.utils.parallelism import ParallelDims
from cosmos_reason1.policy.config import Config as CosmosConfig


class BaseModel(ABC):
    def current_device(self):
        return next(self.parameters()).device

    @cached_property
    def sorted_params(self) -> List[Tuple[str, Tuple[int]]]:
        """
        Returns the state dict of the model and visual model, along with the sorted parameters information.
        The sorted parameters information is a list of tuples, where each tuple contains the parameter name and its shape.
        The state dicts are obtained from the model and visual model respectively.
        """
        sorted_params_info = []
        for k, v in self.named_parameters():
            k = self.map_local_key_to_hf_key(k)
            is_dist_tensor = isinstance(v, torch.distributed.tensor.DTensor)
            local_view = v.to_local() if is_dist_tensor else v
            sorted_params_info.append((k, local_view.shape))
        sorted_params_info.sort(key=lambda x: x[0])
        return sorted_params_info

    @property
    @abstractmethod
    def parallelize_fn(self):
        raise NotImplementedError

    @abstractmethod
    def apply_pipeline_split(self, pp_rank, pp_size):
        raise NotImplementedError

    def post_to_empty_hook(self, cosmos_config: CosmosConfig):
        """
        Hook to be called when the model is moved to CUDA device.
        This is used to re-initialize buffers like `inv_freq` for rotary embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, int]:
        """
        Method to get the position ids of the model.
        This function is declared due to that `Context Parallelism`
        requires the shuffle of both `input_ids` and `position_ids`.

        Args:
            **kwargs: Keyword arguments.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the position ids and the sequence dimension index of the returned position ids.
        """
        raise NotImplementedError

    @abstractmethod
    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: ParallelDims,
        device: torch.device,
    ):
        """
        Load weights from a HuggingFace model.

        Args:
            model_name_or_path (str): The name or path of the model.
            parallel_dims (ParallelDims): The parallel dimensions.
            device (torch.device): The device to load the weights.
        """
        raise NotImplementedError

    @abstractmethod
    def separate_model_parts(self) -> List[torch.nn.Module]:
        """
        Model parts that should be trained in separate optimizers. (i.e. Multi-model training)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def map_local_key_to_hf_key(self, key: str) -> str:
        raise NotImplementedError

    @torch.no_grad()
    def to_hf_state_dict_generator(self, parallel_dims: ParallelDims):
        for name, param in self.named_parameters():
            name = self.map_local_key_to_hf_key(name)
            yield name, param

    @torch.no_grad()
    def maybe_decompose_weights(self, name, param):
        yield name, param

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, max_position_embeddings: Optional[int] = None
    ) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def weight_sync_transform_by_key(
        self, dest_name: str
    ) -> Union[Callable[[], torch.Tensor], torch.Tensor]:
        """
        Get the local view of the tensor from the state dict
        Args:
            name (str): The name of the tensor to be retrieved.
        Returns:
            torch.Tensor: The tensor corresponding to the given name.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_nparams_and_flops(self, seq_len: int) -> tuple[int, int]:
        """
        Get the number of parameters and flops of the model.
        Args:
            seq_len (int): The sequence length of the model.
        Returns:
            tuple[int, int]: The number of parameters and flops of the model.
        """
        raise NotImplementedError
