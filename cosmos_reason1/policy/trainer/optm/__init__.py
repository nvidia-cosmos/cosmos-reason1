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

from typing import List, Dict, Any, TypeVar, Generic, Callable
import functools
import copy
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from cosmos_reason1.policy.config import Config as CosmosConfig
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from cosmos_reason1.utils.logging import logger
import inspect

try:
    from torchao.optim import Adam8bit, AdamW8bit
except ImportError:
    Adam8bit = None
    AdamW8bit = None

T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        optimizer_cls (type[T]): Class of the optimizers.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (List[Dict[str, Any]]): Keyword arguments for the optimizers.
    """

    optimizers: List[T]
    model_parts: List[nn.Module]

    def __init__(
        self,
        optimizer_cls: type[T],
        model_parts: List[nn.Module],
        optimizer_kwargs: List[Dict[str, Any]],
    ) -> None:
        all_params = []
        self.optimizers: List[T] = []
        self.model_parts = model_parts
        for model, optimizer_kwargs_i in zip(self.model_parts, optimizer_kwargs):
            if model is None:
                continue
            params = [p for p in model.parameters() if p.requires_grad]
            self.optimizers.append(optimizer_cls(params, **optimizer_kwargs_i))
            all_params.extend(params)
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Optimizer:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            # Check those grad is None:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        if len(optimizer_kwargs) == 1:
            Optimizer.__init__(self, all_params, optimizer_kwargs[0])
        else:
            # This won't affect the optimizer behavior, since all methods just forward
            # the arguments to the sub-optimizers.
            Optimizer.__init__(
                self,
                all_params,
                {
                    "optimizers_args": optimizer_kwargs,
                },
            )


def build_optimizers(
    model_parts: List[nn.Module],
    config: CosmosConfig,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``job_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
    """
    lr = config.train.optm_lr
    if isinstance(lr, float):
        lr = [lr] * len(model_parts)
    elif isinstance(lr, list):
        assert len(lr) == len(
            model_parts
        ), "The length of lr and model_parts must be the same"
    else:
        raise ValueError(f"Invalid lr: {lr}")

    optm_impl = config.train.optm_impl
    if isinstance(optm_impl, str):
        assert optm_impl in ["fused", "foreach", "for-loop"], "Invalid optm_impl"
        optm_impl = [optm_impl] * len(model_parts)
    elif isinstance(optm_impl, list):
        assert len(optm_impl) == len(
            model_parts
        ), "The length of optm_impl and model_parts must be the same"
        assert all(
            optm_impl_i in ["fused", "foreach", "for-loop"] for optm_impl_i in optm_impl
        ), "Invalid optm_impl"
    else:
        raise ValueError(f"Invalid optm_impl: {optm_impl}")

    fused = [optm_impl_i == "fused" for optm_impl_i in optm_impl]
    foreach = [optm_impl_i == "foreach" for optm_impl_i in optm_impl]

    optimizer_kwargs = [
        {
            "lr": lr_i,
            "eps": config.train.epsilon,
            "betas": config.train.optm_betas,
            "weight_decay": config.train.optm_weight_decay,
            "fused": fused_i,
            "foreach": foreach_i,
        }
        for lr_i, fused_i, foreach_i in zip(lr, fused, foreach)
    ]

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "Adam8bit": Adam8bit,
        "AdamW8bit": AdamW8bit,
    }

    name = config.train.optm_name
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    elif optimizer_classes[name] is None:
        raise NotImplementedError(f"Optimizer {name} not installed.")

    optimizer_cls = optimizer_classes[name]
    init_signature = inspect.signature(optimizer_cls.__init__)
    parameters = init_signature.parameters
    kwarg_names = [
        name
        for name, param in parameters.items()
        if param.default != inspect.Parameter.empty
    ]

    filtered_optimizer_kwargs = []
    for optimizer_kwargs_i in optimizer_kwargs:
        # Filter for kwargs
        optimizer_kwargs_i = {
            k: v for k, v in optimizer_kwargs_i.items() if k in kwarg_names
        }
        # Warn if any kwargs are not used
        unused_kwargs = set(optimizer_kwargs_i.keys()) - set(kwarg_names)
        if unused_kwargs:
            logger.warning(f"Unused kwargs in optimizer-{name}: {unused_kwargs}")
        filtered_optimizer_kwargs.append(optimizer_kwargs_i)
    return OptimizersContainer(optimizer_cls, model_parts, filtered_optimizer_kwargs)


class LRSchedulersContainer(Stateful):
    """Container for multiple learning rate schedulers.

    This class is used to wrap multiple LRSchedulers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.lr_scheduler.LRScheduler``. The design concept is the same as
    ``OptimizersContainer``. This class currently only supports ``LambdaLR``.

    **Note**
    Users who want to customize the lr_scheduler behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same
    signature as ``torch.optim.lr_scheduler.LRScheduler`` class: ``step()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes all the lr schedulers are the same. There is no easy way to support
    resharding for multiple different LRSchedulers because LRScheduler.state_dict() is not
    resharding friendly. Therefore, the limitation is used to allow TorchTitan to support
    lr scheduler resharding.

    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the lr_schedulers.
    """

    schedulers: List[LRScheduler]

    def __init__(self, optimizers: OptimizersContainer, lr_lambda: Callable) -> None:
        assert (
            len(optimizers) > 0
        ), "Must have at least one optimizer to create LRScheduler"

        self.schedulers = [LambdaLR(optimizer, lr_lambda) for optimizer in optimizers]

    def __iter__(self) -> LRScheduler:
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def get_last_lr(self) -> float:
        return self.schedulers[0].get_last_lr()

    def state_dict(self) -> Dict[str, Any]:
        # While there may be multiple schedulers, we only save the first one because
        # the state_dict is the same for all. See the limitations section in the
        # docstring.
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Load the same state_dict for all schedulers. The key value we're concerned
        # within ``LRScheduler.state_dict()`` is ``last_epoch``, which is an integer
        # that is immutable. As long as ``training.steps`` and ``lr_scheduler.warmup_steps``
        # in ``job_config`` remain unchanged when resuming from a checkpoint, this
        # approach is safe. We call ``copy()`` here to ensure extra safety.
        for scheduler in self.schedulers:
            scheduler.load_state_dict(copy.deepcopy(state_dict))


def build_lr_schedulers(
    optimizers: OptimizersContainer, config: CosmosConfig
) -> LRSchedulersContainer:
    """Create a LRSchedulerContainer for the given optimizers and job config.

    This function creates a ``LRSchedulersContainer`` for the given optimizers.
    ``job_config`` should define the correct lr scheduler parameters.

    **Note**
    Users who want to customize the lr scheduler behavior can create their own
    ``LRSchedulersContainer`` subclass and ``build_lr_scheduler``. Passing the
    customized ``build_lr_schedulers`` to ``TrainSpec`` will create the customized
    ``LRSchedulersContainer``.


    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the
            lr_schedulers.
    """
    warmup_steps = int(config.train.optm_warmup_steps)

    def linear_warmup_stable(
        current_step: int,
        warmup_steps: int,
    ):
        if current_step < warmup_steps:
            # linear warmup
            # 0-indexed step, hence + 1 adjustments
            current_step += 1
            curr_adjustment = float(current_step / (warmup_steps + 1))
        else:
            curr_adjustment = 1.0
        return curr_adjustment

    lr_lambda = functools.partial(
        linear_warmup_stable,
        warmup_steps=warmup_steps,
    )
    return LRSchedulersContainer(optimizers, lr_lambda)
