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

from typing import Tuple, Optional, List, Dict, Any
import torch
import torch.distributed.pipelining
from torch.distributed.pipelining import (
    PipelineStage as OriginalPipelineStage,
    Schedule1F1B as OriginalSchedule1F1B,
    ScheduleGPipe as OriginalScheduleGPipe,
)
from torch.distributed.pipelining.stage import (
    _RecvInfo,
    _RootArgPlaceholder,
    _make_tensor_from_meta,
)

from cosmos_reason1.utils.logging import logger


# Common functions for dynamic shape support
def clear_stage(self):
    self._stage_initialized = False
    self._stage.inputs_meta = None
    self._stage._outputs_meta = None


def infer_seq_dim(self, position_ids_shape):
    # Find the dimension index where the current position_ids shape differs from the initial shape
    diff = [
        i
        for i, (a, b) in enumerate(
            zip(position_ids_shape, self.init_position_ids_shape)
        )
        if a != b
    ]
    if len(diff) > 1 and not self.clear_stage_enabled:
        logger.warning(
            "Only one dimension can be different between the current and the initial position ids shape.\
                         Set clear_stage_enabled to True."
        )
        self.clear_stage_enabled = True
        return

    if len(diff) == 1:
        position_ids_seq_dim = diff[0]
        self.position_ids_seq_dim = position_ids_seq_dim
        prev_seq_len = self.init_position_ids_shape[position_ids_seq_dim]
        input_prev_seq_len = prev_seq_len // self.seq_len_multiple
        output_prev_seq_len = (
            input_prev_seq_len if not self._stage.is_last else prev_seq_len
        )

        logger.info(f"Inferred position ids seq dim: {self.position_ids_seq_dim}")
        for input_meta in self._stage.inputs_meta:
            input_meta_shape = input_meta.shape
            logger.info(f"Input meta shape: {input_meta_shape} {input_prev_seq_len=}")
            for i, dim in enumerate(input_meta_shape):
                if dim == input_prev_seq_len:
                    self.input_seq_dim = i
                    logger.info(f"Inferred input meta seq dim: {self.input_seq_dim}")
                    break
            if self.input_seq_dim is None:
                logger.warning(
                    f"Seq dim is not inferred for input meta {input_meta}, set clear_stage_enabled to True."
                )
                self.clear_stage_enabled = True
                break
        for output_meta in self._stage._outputs_meta:
            output_meta_shape = output_meta.shape
            logger.info(
                f"Output meta shape: {output_meta_shape} {output_prev_seq_len=}"
            )
            for i, dim in enumerate(output_meta_shape):
                if dim == output_prev_seq_len:
                    self.output_seq_dim = i
                    logger.info(f"Inferred output meta seq dim: {self.output_seq_dim}")
                    break
            if self.output_seq_dim is None:
                logger.warning(
                    f"Seq dim is not inferred for output meta {output_meta}, set clear_stage_enabled to True."
                )
                self.clear_stage_enabled = True


def update_stage(self, position_ids_shape):
    # TODO: support multiple inputs/outputs meta update
    inputs_meta_len = len(self._stage.inputs_meta)
    outputs_meta_len = len(self._stage._outputs_meta)
    if (inputs_meta_len > 1 or outputs_meta_len > 1) and not self.clear_stage_enabled:
        logger.warning(f"Default pipeline parallelism dynamic shape mode currently only supports at most 1 input meta (got {inputs_meta_len}) \
                         or 1 output meta (got {outputs_meta_len}). Set clear_stage_enabled to True.")
        self.clear_stage_enabled = True

    if (
        self.input_seq_dim is None
        and self.output_seq_dim is None
        and not self.clear_stage_enabled
    ):
        self._infer_seq_dim(position_ids_shape)

    # In case the position_ids seq dim is not inferred, just skip the update
    # e.g. both prev batch and current batch have same seq len
    if self.position_ids_seq_dim is None:
        return

    # clear the stage if clear_stage_enabled is True
    if self.clear_stage_enabled:
        self._clear_stage()
        return

    new_seq_len = position_ids_shape[self.position_ids_seq_dim]
    input_new_seq_len = new_seq_len // self.seq_len_multiple
    output_new_seq_len = input_new_seq_len if not self._stage.is_last else new_seq_len
    logger.debug(
        f"Updating stage with {new_seq_len=} {input_new_seq_len=} {output_new_seq_len=}"
    )
    if inputs_meta_len:
        new_inputs_meta = tuple()
        for input_meta in self._stage.inputs_meta:
            input_meta_shape = list(input_meta.shape)
            if input_meta_shape[self.input_seq_dim] != input_new_seq_len:
                input_meta_shape[self.input_seq_dim] = input_new_seq_len
                new_input_meta = torch.empty(
                    input_meta_shape, device=input_meta.device, dtype=input_meta.dtype
                )
            else:
                new_input_meta = input_meta
            new_inputs_meta += (new_input_meta,)
        self._stage.inputs_meta = new_inputs_meta
        logger.debug(f"Updated inputs meta: {self._stage.inputs_meta}")

    if outputs_meta_len:
        new_outputs_meta = tuple()
        for output_meta in self._stage._outputs_meta:
            output_meta_shape = list(output_meta.shape)
            if output_meta_shape[self.output_seq_dim] != output_new_seq_len:
                output_meta_shape[self.output_seq_dim] = output_new_seq_len
                new_output_meta = torch.empty(
                    output_meta_shape,
                    device=output_meta.device,
                    dtype=output_meta.dtype,
                )
            else:
                new_output_meta = output_meta
            new_outputs_meta += (new_output_meta,)
        self._stage._outputs_meta = new_outputs_meta
        logger.debug(f"Updated outputs meta: {self._stage._outputs_meta}")

    self._stage._reconstruct_forward_infra(self._n_microbatches)
    if self._has_backward:
        self._stage._prepare_backward_infra(self._n_microbatches)


def step_func(self, *args, target=None, losses: Optional[List] = None, **kwargs):
    """
    Run one iteration of the pipeline schedule with *whole-batch* input.
    Will chunk the input into microbatches automatically, and go through the
    microbatches according to the schedule implementation.

    args: positional arguments to the model (as in non-pipeline case).
    kwargs: keyword arguments to the model (as in non-pipeline case).
    target: target for the loss function.
    losses: a list to store the losses for each microbatch.
    """

    # Clean per iteration
    self._stage.clear_runtime_states()

    # Split inputs into microbatches
    args_split, kwargs_split = self._split_inputs(args, kwargs)

    # Split target into microbatches
    if target is not None:
        targets_split = list(torch.tensor_split(target, self._n_microbatches))
    else:
        targets_split = None

    if self.pp_dynamic_shape_enabled is None:
        self.pp_dynamic_shape_enabled = kwargs.get("pp_dynamic_shape_enabled", False)

    position_ids = kwargs_split[0].get("position_ids", None)
    if position_ids is None:
        logger.warning("Position ids is None, set clear_stage_enabled to True")
        self.clear_stage_enabled = True
    else:
        if self.pp_dynamic_shape_enabled and self.init_position_ids_shape is None:
            self.init_position_ids_shape = position_ids.shape
        seq_len_multiple = kwargs.get("seq_len_multiple", None)
        if self.pp_dynamic_shape_enabled and self.seq_len_multiple is None:
            assert (
                seq_len_multiple is not None
            ), "Seq len multiple is required for dynamic shape mode"
            self.seq_len_multiple = seq_len_multiple
    # Update stage if it is already initialized by updating input/output metadata
    # to match the new input shapes and types
    if self.pp_dynamic_shape_enabled and self._stage_initialized:
        self._update_stage(position_ids.shape)

    # Run microbatches
    self._step_microbatches(args_split, kwargs_split, targets_split, losses)

    # Return merged results per original format
    if self._stage.is_last:
        return self._merge_outputs(self._stage.output_chunks)
    else:
        return None


class ScheduleGPipe(OriginalScheduleGPipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Used for input/output metadata update in dynamic shape mode
        self.input_seq_dim = None
        self.output_seq_dim = None
        self.position_ids_seq_dim = None
        self.init_position_ids_shape = None
        self.seq_len_multiple = None
        self.pp_dynamic_shape_enabled = None
        self.first_call = True
        # If true, clear the stage each step
        self.clear_stage_enabled = False

    # Reset the stage initialized flag for dynamic shape support
    def _clear_stage(self):
        clear_stage(self)

    def _infer_seq_dim(self, position_ids_shape):
        infer_seq_dim(self, position_ids_shape)

    def _update_stage(self, position_ids_shape):
        update_stage(self, position_ids_shape)

    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        # training schedule and validation schedule share the same stage, so we need to clear
        # the stage before the first step
        if self.first_call and (
            len(self._stage.inputs_meta) or len(self._stage._outputs_meta)
        ):
            self._clear_stage()
            self.first_call = False

        return step_func(self, *args, target=target, losses=losses, **kwargs)


class Schedule1F1B(OriginalSchedule1F1B):
    def __init__(self, *args, **kwargs):
        if "scale_grads" not in kwargs and torch.__version__ >= "2.7.0":
            # Loss scale is enabled by default in torch 2.7.0
            # We scale the grads manually in this codebase
            kwargs["scale_grads"] = False
        super().__init__(*args, **kwargs)
        # Used for input/output metadata update in dynamic shape mode
        self.input_seq_dim = None
        self.output_seq_dim = None
        self.position_ids_seq_dim = None
        self.init_position_ids_shape = None
        self.seq_len_multiple = None
        self.pp_dynamic_shape_enabled = None
        # If true, clear the stage each step
        self.clear_stage_enabled = False

    # Reset the stage initialized flag for dynamic shape support
    def _clear_stage(self):
        clear_stage(self)

    def _infer_seq_dim(self, position_ids_shape):
        infer_seq_dim(self, position_ids_shape)

    def _update_stage(self, position_ids_shape):
        update_stage(self, position_ids_shape)

    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        return step_func(self, *args, target=target, losses=losses, **kwargs)


class PipelineStage(OriginalPipelineStage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.__version__ >= "2.6.0" and torch.__version__ < "2.7.0":
            self.patched = True
        else:
            self.patched = False

    def backward_one_chunk(self, *args, **kwargs):
        if self.patched:
            return self.cosmos_backward_one_chunk(*args, **kwargs)
        else:
            return super().backward_one_chunk(*args, **kwargs)

    def _reconstruct_forward_infra(
        self,
        num_microbatches: int,
    ) -> Tuple[Any, ...]:
        for chunk_id in range(num_microbatches):
            if not self.is_first:
                # We assume that we always receive from stage - 1
                recv_infos = tuple(
                    [
                        _RecvInfo(
                            f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                            self.stage_index - 1,
                            _make_tensor_from_meta(inp, self.device),
                        )
                        for inp in self.inputs_meta
                    ]
                )
                # In case there is backward pass, set requires_grad for receive buffers
                if self.has_backward:
                    for r in recv_infos:
                        r.buffer.requires_grad_(True)

                self.args_recv_info[chunk_id] = recv_infos
            else:
                self.args_recv_info[chunk_id] = tuple(
                    [_RootArgPlaceholder(i) for i in self.inputs_meta]
                )

        # Send info during forward for each activation
        # only need the rank that is being sent to
        self.act_send_info: Dict[int, List] = {}

        for idx in range(len(self.get_outputs_meta())):
            # We assume we always send to stage + 1
            if not self.is_last:
                self.act_send_info[idx] = [self.stage_index + 1]
            else:
                self.act_send_info[idx] = []

    def cosmos_backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is optional that `dw_runner` was provided to the PipelineStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.

        last_backward is controlled by the schedule and signals synchronization of gradients across DP groups
        after the last backward.
        """
        self._check_chunk_id(bwd_chunk_id)

        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
        else:
            # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads(bwd_chunk_id)
            # If an input to the pipeline requires gradient,
            # `torch.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        grads_input: Tuple[Optional[torch.Tensor], ...] = ()

        # Custom backward function
        if self.dw_builder:
            # TODO: We may want to change our semantics so we are allowed to ignore
            # the 'dw_builder' and call full_backward directly when it is a full_backward op.
            grads_input, _ = self.backward_maybe_with_nosync(
                "full", bwd_kwargs, last_backward=last_backward
            )
            if full_backward:
                self.dw_builder()()
            else:
                self.dw_runner[bwd_chunk_id] = self.dw_builder()
        else:
            if full_backward:
                grads_input, _ = self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward
                )
            else:
                param_groups: List[Dict[str, Any]] | None = None
                # Skip the backward for the first stage since we will perform the weight update with
                # autograd.backward in backward_weight_one_chunk
                if not self.is_first:
                    if isinstance(bwd_kwargs["stage_output"], torch.Tensor):
                        bwd_kwargs["stage_output"] = (bwd_kwargs["stage_output"],)

                    # perform the partial backwards for the inputs with a custom backward function
                    # when the "stage_ouput" is a loss, then it is a tensor, otherwise it is a tuple of tensors
                    grads_input, param_groups = self.backward_maybe_with_nosync(
                        "input", bwd_kwargs, last_backward=last_backward
                    )

                # TODO: we dont need to save this, add to dw_runner?
                self.backward_state[bwd_chunk_id] = (
                    bwd_kwargs["input_values"],
                    param_groups,
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                )
                # Save a placeholder for the dw_runner
                self.dw_runner[bwd_chunk_id] = lambda: None

        self.bwd_cache[bwd_chunk_id] = grads_input

        if self.is_last and not self.is_first:
            # Autograd dependencies:
            #    rest_of_autograd_graph -> stage_output -> loss
            # stage_output is no longer used in the last stage for backward and only needed
            # to return to the user in merge_output_chunks, therefore
            # this should be detached to release autograd graph context and free memory earlier
            for t in stage_output:
                if not t._is_view():
                    t.detach_()
