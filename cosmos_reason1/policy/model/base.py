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
from cosmos_reason1.dispatcher.data.packer.base import DataPacker
from transformers import AutoConfig


class BaseModel(ABC):
    def current_device(self):
        """
        Get the current device of the model
        """
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

    @torch.no_grad()
    def maybe_decompose_weights_to_hf_naming(self, name, param):
        """
        Decompose the weights of the model parameters into fine-grained weights
        This is especially useful for models with non-symmetric parameter layout than the original HuggingFace one
        For example, MoE experts' weights are stacked in the 0th dimension,
        while they are stored in different keys in the original HuggingFace naming convention
        """
        yield name, param

    def tensor_precollect_required_for_sync(self, name: str) -> bool:
        """
        Check if the tensor sync precollect is required for the given name.
        Args:
            name (str): The name of the tensor.
        Returns:
            bool: True if the tensor sync precollect is required, False otherwise.
        """
        return False

    """
    Abstract methods
    """

    @staticmethod
    @abstractmethod
    def supported_model_types():
        raise NotImplementedError

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
    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Method to get the position ids of the model.
        This function is declared due to that `Context Parallelism`
        requires the shuffle of both `input_ids` and `position_ids`.

        Args:
            **kwargs: Keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]:
                - Tensor of position ids
                - Tensor of input ids
                - Sequence dimension index of position ids.
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

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def weight_sync_transform_by_key(
        cls, dest_name: str
    ) -> Union[Callable[[], torch.Tensor], torch.Tensor]:
        """
        Get the local view of the tensor from the state dict
        Args:
            name (str): The name of the tensor to be retrieved.
        Returns:
            torch.Tensor: The tensor corresponding to the given name.
        """
        raise NotImplementedError

    @cached_property
    def weight_sync_transforms(self) -> List[Tuple[str, Tuple[int], torch.Tensor]]:
        """
        Get the local view of the tensors from the state dict.
        This method retrieves the state dict of the model, clears the weight names,
        and returns a list of tuples containing the destination name, shape, and local view of each tensor.
        Returns:
            List[Tuple[str, Tuple[int], torch.Tensor]]: A list of tuples containing the destination name,
            shape, and local view of each tensor.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_nparams_and_flops(cls, seq_len: int) -> tuple[int, int]:
        """
        Get the number of parameters and flops of the model.
        Args:
            seq_len (int): The sequence length of the model.
        Returns:
            tuple[int, int]: The number of parameters and flops of the model.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def data_packer(cls) -> DataPacker:
        raise NotImplementedError
