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

from cosmos_reason1.policy.trainer import Trainer
from cosmos_reason1.utils.parallelism import (
    ParallelDims,
    create_context_parallel_ctx,
)
from cosmos_reason1.policy.config import (
    Config as CosmosConfig,
    SFTDataConfig,
    config_hash,
)
from cosmos_reason1.utils.util import compute_mfu
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.wandb_logger import is_wandb_available, log_wandb
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import cosmos_reason1.utils.util as util
import cosmos_reason1.utils.distributed as dist_util
import cosmos_reason1.utils.cache as cache
from transformers import AutoTokenizer
from datasets import concatenate_datasets
from cosmos_reason1.dispatcher.data.packer.base import DataPacker
import functools
import os
from typing import Optional, Dict, Any
from tqdm import tqdm


def collate_fn(
    batch,
    pad_token_id,
    data_packer: DataPacker,
    config: CosmosConfig,
    seq_len_multiple=1,
    ignore_label_id=-100,
    fixed_length: Optional[int] = None,
):
    if fixed_length is None:
        max_len = min(
            config.policy.model_max_length,
            data_packer.sft_compute_max_len(batch),
        )
    else:
        max_len = fixed_length

    if seq_len_multiple > 1:
        max_len = (
            (max_len + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple
        )

    model_input: Dict[str, Any] = data_packer.sft_collate_fn(
        batch,
        computed_max_len=max_len,
        pad_token_id=pad_token_id,
        ignore_label_id=ignore_label_id,
    )

    return model_input


def construct_dataset(
    config: SFTDataConfig,
    tokenizer: AutoTokenizer,
    data_packer: DataPacker,
    user_provided_dataset: Optional[Dataset] = None,
):
    if user_provided_dataset is not None:
        dataset = None
        train_dataset = user_provided_dataset
        logger.info("Using user-provided dataset, which will skip split processing.")
    else:
        dataset = util.load_data_from_disk_or_hf(
            config.dataset.name,
            config.dataset.subset,
            config.dataset.revision or None,
        )
        dataset_list = []
        for split_name in config.dataset.train_split:
            logger.info(
                f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
            )
            dataset_list.append(dataset[split_name])
        train_dataset = concatenate_datasets(dataset_list)
    logger.info(f"Final dataset size = {len(train_dataset)}")

    try:
        if dataset is not None:
            dataset_list = []
            for split_name in config.dataset.test_split:
                dataset_list.append(dataset[split_name])
            test_dataset = concatenate_datasets(dataset_list)
            if len(test_dataset) == 0:
                raise ValueError("Test dataset is empty")
        else:
            raise ValueError("Test dataset is empty")
    except Exception:
        if isinstance(train_dataset, torch.utils.data.Dataset):
            # Define the split ratio (e.g., 80% train, 20% test)
            if config.dataset.test_size is None:
                logger.warning(
                    "No test size specified, using 10% of the training dataset for testing."
                )
                config.dataset.test_size = 0.1
            if isinstance(config.dataset.test_size, float):
                n_test_samples = int(len(train_dataset) * config.dataset.test_size)
            else:
                n_test_samples = config.dataset.test_size
            n_test_samples = max(min(n_test_samples, len(train_dataset) - 1), 1)

            # Generate deterministic indices
            indices = list(range(len(train_dataset)))
            test_indices = indices[:n_test_samples]
            train_indices = indices[n_test_samples:]

            test_dataset = torch.utils.data.Subset(train_dataset, test_indices)
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        else:
            assert hasattr(
                train_dataset, "train_test_split"
            ), "train_dataset must have train_test_split method"
            train_test_split = train_dataset.train_test_split(
                test_size=config.dataset.test_size, shuffle=False
            )
            train_dataset = train_test_split["train"]
            test_dataset = train_test_split["test"]

    train_sft_dataset = SFTDataset(
        config,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_packer=data_packer,
        is_user_dataset=user_provided_dataset is not None,
    )
    test_sft_dataset = SFTDataset(
        config,
        tokenizer=tokenizer,
        dataset=test_dataset,
        data_packer=data_packer,
        is_user_dataset=user_provided_dataset is not None,
    )

    return train_sft_dataset, test_sft_dataset


class SFTDataset(Dataset):
    def __init__(
        self,
        config: SFTDataConfig,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        data_packer: DataPacker,
        is_user_dataset: bool = False,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.column_name = config.conversation_column_name
        self.dataset = dataset
        self.data_packer = data_packer
        self.is_user_dataset = is_user_dataset
        self.cache = None
        if self.config.enable_dataset_cache:
            # TODO(zjx): can we reuse the cache between different training jobs?
            # It's not stable yet, we only checked if the config is the same
            # If there are any problems, it is recommended that the user clears the cache folder
            cache_folder = os.path.join(
                os.environ.get(
                    "COSMOS_CACHE",
                    os.path.join(os.path.expanduser("~"), ".cache/cosmos/"),
                ),
                "datasets_cache",
                f"{self.config.dataset.name}-{config_hash(config)}",
            )
            logger.info(f"SFTDataset Cache folder: {cache_folder}")
            self.cache = cache.DiskCache(cache_folder)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # we only cache on_the_fly result
        if self.cache is not None:
            cache_obj = self.cache.get(idx)
            if cache_obj is not None:
                return cache_obj

        raw_item = (
            self.dataset[idx][self.column_name]
            if not self.is_user_dataset and self.column_name
            else self.dataset[idx]
        )

        item: Dict[str, Any] = self.data_packer.sft_process_sample(raw_item)

        if self.cache is not None:
            # try cache obj
            self.cache.set(idx, item)
        return item


class SFTTrainer(Trainer):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims):
        super(SFTTrainer, self).__init__(config, parallel_dims)

        # Enlarge the compile cache size for validation
        if config.train.compile and config.train.enable_validation:
            torch._dynamo.config.cache_size_limit = 64

        self.dp_rank, self.dp_world_size = 0, 1
        if parallel_dims.dp_enabled:
            self.dp_rank = parallel_dims.mesh["dp"].get_local_rank()
            self.dp_world_size = parallel_dims.mesh["dp"].size()
        train_dataset, val_dataset = construct_dataset(
            config.train.train_policy,
            tokenizer=self.tokenizer,
            data_packer=self.data_packer,
            user_provided_dataset=self.sft_user_dataset,
        )
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=self.dp_world_size, rank=self.dp_rank
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=self.dp_world_size, rank=self.dp_rank
        )

        assert (
            self.tokenizer.pad_token_id is not None
        ), "Tokenizer must have a pad token id"
        self.train_data_loader = DataLoader(
            train_dataset,
            batch_size=config.train.train_batch_per_replica,
            shuffle=config.train.train_policy.dataloader_shuffle,
            num_workers=config.train.train_policy.dataloader_num_workers,
            prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
            sampler=train_sampler,
            collate_fn=functools.partial(
                collate_fn,
                pad_token_id=self.tokenizer.pad_token_id,
                seq_len_multiple=self.seq_len_multiple,
                fixed_length=config.policy.model_max_length
                if parallel_dims.pp_enabled and not parallel_dims.pp_dynamic_shape
                else None,
                data_packer=self.data_packer,
                config=config,
            ),
        )
        self.val_data_loader = DataLoader(
            val_dataset,
            batch_size=config.train.validation_batch_per_replica,
            num_workers=config.train.train_policy.dataloader_num_workers,
            prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
            sampler=val_sampler,
            collate_fn=functools.partial(
                collate_fn,
                pad_token_id=self.tokenizer.pad_token_id,
                seq_len_multiple=self.seq_len_multiple,
                fixed_length=config.policy.model_max_length
                if parallel_dims.pp_enabled and not parallel_dims.pp_dynamic_shape
                else None,
                data_packer=self.data_packer,
                config=config,
            ),
        )
        # For iteration control
        self.epoch = config.train.epoch
        self.total_steps = (
            len(self.train_data_loader) * self.epoch // self.dp_world_size
        )
        self.train_step = 0

        # Load model
        if config.train.resume:
            try:
                ckpt_extra_vars = self.ckpt_manager.load_checkpoint(
                    model=self.model,
                    optimizer=self.optimizers,
                    scheduler=self.lr_schedulers,
                )
                ckpt_total_steps = ckpt_extra_vars.get("total_steps", 0)
                if ckpt_total_steps != self.total_steps:
                    logger.warning(
                        f"Checkpoint total steps {ckpt_total_steps} does not match expected {self.total_steps}. Start training from step 0"
                    )
                else:
                    self.train_step = ckpt_extra_vars.get("step", 0)
            except Exception:
                logger.error(
                    f"Cannot resume from {self.config.train.resume}. Trying to load from HuggingFace..."
                )
                self.model.load_hf_weights(
                    config.policy.model_name_or_path, parallel_dims, self.device
                )
        else:
            self.model.load_hf_weights(
                config.policy.model_name_or_path, parallel_dims, self.device
            )
        self.model.train()

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def validate(self):
        logger.info(f"Validation at step {self.train_step}/{self.total_steps}...")
        self.model.eval()
        with torch.no_grad():
            val_total_loss = 0.0
            for val_batch in tqdm(self.val_data_loader, desc="Validation"):
                for k, v in val_batch.items():
                    val_batch[k] = (
                        v.to(self.device) if isinstance(v, torch.Tensor) else v
                    )
                val_inputs = val_batch["input_ids"]
                val_labels = val_batch.pop("label_ids")
                val_position_ids, _, val_pos_seq_dim = self.model.get_position_ids(
                    **val_batch
                )

                val_cp_context = (
                    create_context_parallel_ctx(
                        cp_mesh=self.parallel_dims.mesh["cp"],
                        cp_buffers=[val_inputs, val_labels, val_position_ids],
                        cp_seq_dims=[1, 1, val_pos_seq_dim],
                        cp_no_restore_buffers={
                            val_inputs,
                            val_labels,
                            val_position_ids,
                        },
                        cp_rotate_method=self.config.parallelism.cp_rotate_method,
                    )
                    if self.parallel_dims.cp_enabled
                    else None
                )

                with self.context(val_cp_context):
                    if self.parallel_dims.pp_enabled:
                        pp_last_stage = (
                            self.parallel_dims.pp_coord[0]
                            == self.parallel_dims.pp_coord[1] - 1
                        )
                        pp_first_stage = self.parallel_dims.pp_coord[0] == 0

                        if pp_first_stage:
                            self.pp_scheduler_val.step(
                                **val_batch,
                                position_ids=val_position_ids,
                                pp_dynamic_shape_enabled=self.parallel_dims.pp_dynamic_shape_enabled,
                                seq_len_multiple=self.seq_len_multiple,
                            )
                        else:
                            pp_out = self.pp_scheduler_val.step(
                                position_ids=val_position_ids,
                                pp_dynamic_shape_enabled=self.parallel_dims.pp_dynamic_shape_enabled,
                                seq_len_multiple=self.seq_len_multiple,
                            )

                        if pp_last_stage:
                            val_logits = pp_out[:, :-1].contiguous()
                            val_loss = self.loss_fn(
                                val_logits.view(-1, val_logits.size(-1)),
                                val_labels[:, 1:].contiguous().view(-1),
                            )
                        else:
                            val_loss = torch.tensor([-1.0], device=self.device)
                    else:
                        val_logits = self.model(
                            **val_batch, position_ids=val_position_ids
                        )[:, :-1].contiguous()
                        val_loss = self.loss_fn(
                            val_logits.view(-1, val_logits.size(-1)),
                            val_labels[:, 1:].contiguous().view(-1),
                        )
                    val_total_loss += val_loss.item() * val_inputs.size(0)
            val_avg_loss = val_total_loss / len(self.val_data_loader.dataset)
            logger.info(f"Validation loss: {val_avg_loss}")
        return val_avg_loss

    def train(self):
        self.profiler.start()
        pp_last_stage = False

        start_epoch = 0
        data_loader_bias = 0
        # Resume training from the last checkpoint if needed
        if self.config.train.resume and self.train_step > 0:
            logger.info(
                f"Resuming training from step {self.train_step}/{self.total_steps}..."
            )
            start_epoch = self.train_step // len(self.train_data_loader)
            data_loader_bias = self.train_step % len(self.train_data_loader)

        for cur_epoch in range(start_epoch, self.epoch):
            logger.info(f"Training epoch {cur_epoch + 1}/{self.epoch}")
            for batch in self.train_data_loader:
                if data_loader_bias > 0:
                    data_loader_bias -= 1
                    continue
                self.model.train()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for k, v in batch.items():
                    batch[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v

                inputs = batch["input_ids"]
                labels = batch.pop("label_ids")

                position_ids, input_ids, pos_seq_dim = self.model.get_position_ids(
                    **batch
                )

                self.optimizers.zero_grad()

                cp_context = (
                    create_context_parallel_ctx(
                        cp_mesh=self.parallel_dims.mesh["cp"],
                        cp_buffers=[inputs, labels, position_ids],
                        cp_seq_dims=[1, 1, pos_seq_dim],
                        cp_no_restore_buffers={inputs, labels, position_ids},
                        cp_rotate_method=self.config.parallelism.cp_rotate_method,
                    )
                    if self.parallel_dims.cp_enabled
                    else None
                )

                with self.context(cp_context):
                    if self.parallel_dims.pp_enabled:
                        pp_last_stage = (
                            self.parallel_dims.pp_coord[0]
                            == self.parallel_dims.pp_coord[1] - 1
                        )
                        pp_first_stage = self.parallel_dims.pp_coord[0] == 0

                        # Pipeline Parallel forward / backward inside step() call
                        targets, losses = (
                            (labels, []) if pp_last_stage else (None, None)
                        )
                        if pp_first_stage:
                            self.pp_scheduler.step(
                                **batch,
                                position_ids=position_ids,
                                pp_dynamic_shape_enabled=self.parallel_dims.pp_dynamic_shape_enabled,
                                seq_len_multiple=self.seq_len_multiple,
                            )
                        else:
                            # FWD + BWD if it is 1F1B-like scheduler
                            self.pp_scheduler.step(
                                position_ids=position_ids,
                                target=targets,
                                losses=losses,
                                pp_dynamic_shape_enabled=self.parallel_dims.pp_dynamic_shape_enabled,
                                seq_len_multiple=self.seq_len_multiple,
                            )
                        loss = (
                            torch.mean(torch.stack(losses)).to(self.device)
                            if pp_last_stage
                            else torch.tensor([-1.0], device=self.device)
                        )
                    else:
                        logits = self.model(**batch, position_ids=position_ids)[
                            :, :-1
                        ].contiguous()
                        loss = self.loss_fn(
                            logits.view(-1, logits.size(-1)),
                            labels[:, 1:].contiguous().view(-1),
                        )
                        loss.backward()
                loss = loss.detach()

                for model_part in self.model_parts:
                    """
                    Do gradient clipping in group for unsymmetric sharding compatibility
                    """
                    dist_util.gradient_norm_clipping(
                        # Must pass empty list even if model_part is None,
                        # GradNorm across pp stages will fail if some rank does not join the barrier
                        [p for p in model_part.parameters()]
                        if model_part is not None
                        else [],
                        self.config.train.optm_grad_norm_clip,
                        foreach=True,
                        pp_mesh=self.parallel_dims.mesh["pp"]
                        if self.parallel_dims.pp_enabled
                        else None,
                    )

                self.optimizers.step()
                self.lr_schedulers.step()

                self.train_step += 1
                end_event.record()
                self.train_event_queue.append((start_event, end_event, self.train_step))

                if (
                    self.parallel_dims.dp_replicate_enabled
                    or self.parallel_dims.dp_shard_enabled
                    or self.parallel_dims.cp_enabled
                ):
                    global_avg_loss, global_max_loss = (  # noqa: F841
                        dist_util.dist_mean(loss, self.parallel_dims.mesh["dp_cp"]),
                        dist_util.dist_max(loss, self.parallel_dims.mesh["dp_cp"]),
                    )
                else:
                    global_avg_loss = global_max_loss = loss.item()  # noqa: F841

                if self.config.logging.logger:
                    if util.is_master_rank(self.parallel_dims, self.global_rank):
                        while (
                            len(self.train_event_queue) > 0
                            and self.train_event_queue[0][1].query()
                        ):
                            iter_start_event, iter_end_event, iter_step = (
                                self.train_event_queue.popleft()
                            )
                            iter_time = (
                                iter_start_event.elapsed_time(iter_end_event) / 1000.0
                            )  # in seconds
                            if (
                                "wandb" in self.config.logging.logger
                                and is_wandb_available()
                            ):
                                log_wandb(
                                    run=self.wandb_run,
                                    data={
                                        "train/iteration_time": iter_time,
                                    },
                                    step=iter_step,
                                )
                            if "console" in self.config.logging.logger:
                                logger.info(
                                    f"Step: {iter_step}/{self.total_steps}, Iteration time: {iter_time:.3f}s."
                                )

                        report_data = {
                            "train/loss_avg": global_avg_loss,
                            "train/loss_max": global_max_loss,
                            "train/learning_rate": self.lr_schedulers.get_last_lr()[0],
                        }

                        # FIXME(dinghaoy): only compute MFU of rank 0, if enable tp or pp,
                        # it will be inaccurate. Need a reduce for all the metrics.
                        if self.config.logging.report_mfu:
                            mfu = compute_mfu(
                                model=self.model,
                                n_tokens=np.prod(input_ids.shape),
                                iter_time=iter_time,
                                num_gpus=self.world_size,
                                dtype=self.config.train.param_dtype,
                            )
                            for k, v in mfu.items():
                                report_data[f"train/{k}"] = v
                        if (
                            "wandb" in self.config.logging.logger
                            and is_wandb_available()
                        ):
                            log_wandb(
                                run=self.wandb_run,
                                data=report_data,
                                step=self.train_step,
                            )
                        if "console" in self.config.logging.logger:
                            logger.info(
                                f"Step: {self.train_step}/{self.total_steps}, Loss: {global_avg_loss:.5f}, Learning rate: {self.lr_schedulers.get_last_lr()[0]:.5e}."
                            )

                # For profiling
                self.profiler.step()

                val_score = None
                # validation
                if (
                    self.config.train.enable_validation
                    and self.train_step % self.config.train.validation_freq == 0
                ):
                    val_score = self.validate()

                # save checkpoint
                if (
                    self.config.train.ckpt.enable_checkpoint
                    and self.train_step % self.config.train.ckpt.save_freq == 0
                    and self.train_step > 0
                ):
                    # TODO(dinghaoy): support export safetensors asynchronously.
                    if self.config.train.ckpt.export_safetensors:
                        logger.info(
                            f"Saving huggingface checkpoint at step {self.train_step} to {self.config.train.output_dir}..."
                        )
                        self.export_safetensors(
                            output_dir=self.config.train.output_dir,
                            rel_path=os.path.join(
                                "safetensors",
                                f"step_{self.train_step}",
                            ),
                            trainable_only=False,
                        )
                    logger.info(
                        f"Saving cosmos checkpoint at step {self.train_step}..."
                    )
                    self.ckpt_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizers,
                        scheduler=self.lr_schedulers,
                        step=self.train_step,
                        total_steps=self.total_steps,
                    )
                    self.ckpt_manager.save_check(
                        step=self.train_step,
                        val_score=val_score,
                        pp_enabled=self.parallel_dims.pp_enabled,
                        pp_last_stage=pp_last_stage,
                        pp_master_rank=self.parallel_dims.world_size
                        - self.parallel_dims.world_size / self.parallel_dims.pp,
                    )

        # process the final step
        if self.config.train.enable_validation:
            val_score = self.validate()
        if self.config.train.ckpt.export_safetensors:
            logger.info(
                f"Saving final huggingface checkpoint to {self.config.train.output_dir}..."
            )
            self.export_safetensors(
                output_dir=self.config.train.output_dir,
                rel_path=os.path.join(
                    "safetensors",
                    f"step_{self.train_step}",
                ),
                trainable_only=False,
                is_final=True,
            )
        if self.config.train.ckpt.enable_checkpoint:
            logger.info(
                f"Training finished at step {self.train_step}/{self.total_steps}, saving final cosmos checkpoint..."
            )
            self.ckpt_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizers,
                scheduler=self.lr_schedulers,
                step=self.train_step,
                total_steps=self.total_steps,
                is_final=True,
            )
            self.ckpt_manager.save_check(
                step=self.train_step,
                val_score=val_score,
                pp_enabled=self.parallel_dims.pp_enabled,
                pp_last_stage=pp_last_stage,
                pp_master_rank=self.parallel_dims.world_size
                - self.parallel_dims.world_size / self.parallel_dims.pp,
            )

    @property
    def pp_loss_fn(self):
        def cross_entropy_loss(
            output: torch.Tensor, target: torch.LongTensor
        ) -> torch.Tensor:
            """Common cross-entropy loss function for Transformer models training."""
            return torch.nn.functional.cross_entropy(
                output[:, :-1].flatten(0, 1).float(), target[:, 1:].flatten(0, 1)
            )

        return torch.compile(cross_entropy_loss)
