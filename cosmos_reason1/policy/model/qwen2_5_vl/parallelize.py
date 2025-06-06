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

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)
from cosmos_reason1.utils.parallelism import ParallelDims
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.policy.config import Config as CosmosConfig
from cosmos_reason1.patch import PipelineStage, Schedule1F1B, ScheduleGPipe
import os
from typing import Callable, Optional


def parallelize(
    model: nn.Module,
    parallel_dims: ParallelDims,
    config: CosmosConfig,
    pp_loss_fn: Optional[Callable] = None,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    world_mesh = parallel_dims.mesh

    pipeline_parallelize(model, parallel_dims, config)

    if parallel_dims.tp_enabled:
        apply_tp(
            model,
            world_mesh["tp"],
            enable_float8_tensorwise_tp=False,
            enable_async_tp=config.train.async_tp_enabled,
            parallel_dims=parallel_dims,
        )

    if config.policy.model_gradient_checkpointing:
        apply_ac(model)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if config.train.compile:
        """
        Why we need to apply compile after AC wrapping and before FSDP?
        https://github.com/pytorch/torchtitan/issues/472#issuecomment-2242200809
        """
        apply_compile(model)

    reshard_after_forward_policy = config.train.fsdp_reshard_after_forward
    # For visual model, TP mesh should be merged into DP_Shard
    if model.visual is not None and (
        parallel_dims.tp_enabled
        or parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
    ):
        logger.info(
            f"Applying FSDP(TP-merged: {parallel_dims.tp_enabled}) to the visual model"
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=config.train.param_torch_dtype,
            reduce_dtype=config.train.fsdp_reduce_torch_dtype,
        )
        fsdp_config = {"mesh": world_mesh["dp_cp_tp"], "mp_policy": mp_policy}
        if config.train.fsdp_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()
        for layer_id, transformer_block in model.visual.blocks.items():
            if reshard_after_forward_policy == "always":
                reshard_after_forward = True
            elif reshard_after_forward_policy == "never":
                reshard_after_forward = False
            elif reshard_after_forward_policy == "default":
                if parallel_dims.pp_enabled:
                    # For PP, do not reshard after forward to avoid per-microbatch
                    # all-gathers, which can be expensive and non-overlapped
                    reshard_after_forward = False
                else:
                    # As an optimization, do not reshard after forward for the last
                    # transformer block since FSDP would prefetch it immediately
                    reshard_after_forward = int(layer_id) < len(model.visual.blocks) - 1
            else:
                raise ValueError(
                    f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
                )
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        fully_shard(
            model.visual,
            **fsdp_config,
            reshard_after_forward=not parallel_dims.pp_enabled,
        )

    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        elif parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
            dp_mesh_dim_names = ("dp_shard_cp",)
        else:
            dp_mesh_dim_names = ()
        # apply FSDP or HSDP, potentially with Context Parallel
        apply_fsdp(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=config.train.param_torch_dtype,
            reduce_dtype=config.train.fsdp_reduce_torch_dtype,
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=config.train.fsdp_offload,
            reshard_after_forward_policy=reshard_after_forward_policy,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if config.train.fsdp_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=config.train.compile,
            enable_compiled_autograd=config.train.compile,
        )

    pp_rank, pp_size = parallel_dims.pp_coord
    if pp_size > 1:
        proc_group = parallel_dims.mesh["pp"].get_group()
        # IMPORTANT:
        # Only CUDA device is acceptable for PP Schedule, `meta` device will cause hanging issue
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

        stage = PipelineStage(
            model,
            pp_rank,
            pp_size,
            device,
            group=proc_group,
        )

        assert (
            config.train.train_batch_per_replica
            % config.policy.parallelism.pp_micro_batch_size
            == 0
        ), "train_batch must be divisible by pp_micro_batch_size"
        assert (
            (
                config.train.train_batch_per_replica
                // config.policy.parallelism.pp_micro_batch_size
            )
            % pp_size
            == 0
        ), "train_batch / pp_micro_batch_size must be divisible by pp_size"
        assert pp_loss_fn is not None, "pp_loss_fn must be provided"
        n_microbatches = (
            config.train.train_batch_per_replica
            // config.policy.parallelism.pp_micro_batch_size
        )
        if config.train.enable_validation:
            assert (
                config.train.validation_batch_per_replica
                % config.policy.parallelism.pp_micro_batch_size
                == 0
            ), "validation_batch must be divisible by pp_micro_batch_size"
            assert (
                (
                    config.train.validation_batch_per_replica
                    // config.policy.parallelism.pp_micro_batch_size
                )
                % pp_size
                == 0
            ), "validation_batch / pp_micro_batch_size must be divisible by pp_size"
            n_val_microbatches = (
                config.train.validation_batch_per_replica
                // config.policy.parallelism.pp_micro_batch_size
            )

        if config.train.train_policy.type == "grpo":
            assert (
                config.train.train_batch_per_replica
                % (
                    config.policy.parallelism.dp_shard_size
                    * config.train.train_policy.mini_batch
                )
                == 0
            ), "train_batch must be divisible by dp_shard_size * mini_batch"
            assert (
                config.train.train_policy.mini_batch
                % config.policy.parallelism.pp_micro_batch_size
                == 0
            ), "mini_batch must be divisible by pp_micro_batch_size"
            n_microbatches = (
                config.train.train_policy.mini_batch
                // config.policy.parallelism.pp_micro_batch_size
            )

        schedule = Schedule1F1B(
            stage=stage,
            n_microbatches=n_microbatches,
            loss_fn=pp_loss_fn,
        )
        if config.train.enable_validation:
            val_schedule = ScheduleGPipe(
                stage=stage,
                n_microbatches=n_val_microbatches,
            )
            return schedule, val_schedule
        else:
            return schedule, None
    else:
        return None, None


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
    parallel_dims: ParallelDims,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    tp_plan = {
        "embed_tokens": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        "norm": SequenceParallel(),
        "lm_head": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate(),
            use_local_output=True,
        ),
    }
    if parallel_dims.pp_coord[0] == 0:
        tp_plan["identity_layer"] = PrepareModuleOutput(
            output_layouts=Replicate(),
            desired_output_layouts=Shard(1),
            use_local_output=True,
        )

    parallelize_module(
        model.model,
        tp_mesh,
        tp_plan,
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for layer_id, transformer_block in model.model.layers.items():
        layer_plan = {
            "input_layernorm": SequenceParallel(),
            "self_attn": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "self_attn.q_proj": colwise_parallel(),
            "self_attn.k_proj": colwise_parallel(),
            "self_attn.v_proj": colwise_parallel(),
            "self_attn.o_proj": rowwise_parallel(output_layouts=Shard(1)),
            "post_attention_layernorm": SequenceParallel(),
            "mlp": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.gate_proj": colwise_parallel(),
            "mlp.down_proj": rowwise_parallel(output_layouts=Shard(1)),
            "mlp.up_proj": colwise_parallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )


# for selective op activation checkpointing
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
}


def _apply_ac_to_transformer_block(module: nn.Module):
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    def _get_custom_policy(meta):
        def _custom_policy(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                meta[mm_count_key] += 1
            # Saves output of all compute ops, except every second mm
            to_save = func in _save_list and not (
                func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta = defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return ptd_checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=False,
    )


def apply_ac(model: nn.Module):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(transformer_block)
        model.model.layers.register_module(layer_id, transformer_block)

    # ``model.visual`` could get deleted by pipeline split
    if model.visual is not None:
        for layer_id, transformer_block in model.visual.blocks.items():
            transformer_block = _apply_ac_to_transformer_block(transformer_block)
            model.visual.blocks.register_module(layer_id, transformer_block)

    logger.info("Applied activation checkpointing to the model")


def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.model.layers.named_children():
        transformer_block = torch.compile(transformer_block, fullgraph=True)
        model.model.layers.register_module(layer_id, transformer_block)

    # ``model.visual`` could get deleted by pipeline split
    if model.visual is not None:
        for layer_id, transformer_block in model.visual.blocks.named_children():
            transformer_block = torch.compile(transformer_block, fullgraph=True)
            model.visual.blocks.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    for layer_id, transformer_block in model.model.layers.items():
        if reshard_after_forward_policy == "always":
            reshard_after_forward = True
        elif reshard_after_forward_policy == "never":
            reshard_after_forward = False
        elif reshard_after_forward_policy == "default":
            if pp_enabled:
                # For PP, do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                reshard_after_forward = False
            else:
                # As an optimization, do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately
                reshard_after_forward = int(layer_id) < len(model.model.layers) - 1
        else:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    if model.model.embed_tokens is not None:
        fully_shard(model.model.embed_tokens, **fsdp_config, reshard_after_forward=True)
    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")


def pipeline_parallelize(
    model: nn.Module,
    parallel_dims: ParallelDims,
    config: CosmosConfig,
):
    if parallel_dims.pp_enabled:
        pp_rank, pp_size = parallel_dims.pp_coord
        model.apply_pipeline_split(pp_rank, pp_size)
    else:
        logger.info("Pipeline is not enabled, skipping")
