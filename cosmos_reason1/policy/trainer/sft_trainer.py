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
from cosmos_reason1.utils.util import compute_mfu, basename_from_modelpath
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.wandb_logger import is_wandb_available, log_wandb
import time
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import cosmos_reason1.utils.util as util
import cosmos_reason1.utils.distributed as dist_util
import cosmos_reason1.utils.cache as cache
from transformers import AutoTokenizer, AutoConfig, AutoProcessor
from datasets import concatenate_datasets
from qwen_vl_utils import process_vision_info
import functools
import os
import copy
from typing import Optional
from tqdm import tqdm
import torch.nn.functional as F


def collate_fn(
    batch,
    pad_token_id,
    seq_len_multiple=1,
    ignore_label_id=-100,
    vision_ids=None,
    fixed_length: Optional[int] = None,
):
    max_len = (
        max([len(x["input_ids"]) for x in batch])
        if fixed_length is None
        else fixed_length
    )
    if seq_len_multiple > 1:
        max_len = (
            (max_len + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple
        )

    input_ids = torch.tensor(
        [
            x["input_ids"] + [pad_token_id] * (max(0, max_len - len(x["input_ids"])))
            for x in batch
        ],
        dtype=torch.long,
    )
    label_ids = torch.tensor(
        [
            x["input_ids"] + [ignore_label_id] * (max(0, max_len - len(x["input_ids"])))
            for x in batch
        ],
        dtype=torch.long,
    )

    # No loss on the vision tokens, since they are not language modal
    if vision_ids is not None:
        assert isinstance(vision_ids, list)
        for vision_id in vision_ids:
            if vision_id is not None:
                label_ids[label_ids == vision_id] = ignore_label_id

    pixel_values_videos = []
    video_grid_thw = []
    second_per_grid_ts = []
    pixel_values_images = []
    image_grid_thw = []
    pixel_values_videos_lengths_per_sample = []
    pixel_values_images_lengths_per_sample = []
    for x in batch:
        if "pixel_values_videos" in x:
            pixel_values_videos.append(x["pixel_values_videos"])
            video_grid_thw.append(x["video_grid_thw"])
            second_per_grid_ts.append(x["second_per_grid_ts"])
            pixel_values_videos_lengths_per_sample.append(
                x["pixel_values_videos_lengths_per_sample"]
            )
        if "pixel_values_images" in x:
            pixel_values_images.append(x["pixel_values_images"])
            image_grid_thw.append(x["image_grid_thw"])
            pixel_values_images_lengths_per_sample.append(
                x["pixel_values_images_lengths_per_sample"]
            )

    if len(pixel_values_videos) > 0:
        # pixel_values_videos = torch.cat(pixel_values_videos, dim=0)
        max_len = max([x.shape[0] for x in pixel_values_videos])
        for i in range(len(pixel_values_videos)):
            pixel_values_videos[i] = pixel_values_videos[i].unsqueeze(0)
            assert (
                pixel_values_videos[i].ndim == 3
            ), f"pixel_values_videos[i].ndim: {pixel_values_videos[i].ndim}"
            pixel_values_videos[i] = F.pad(
                pixel_values_videos[i],
                (0, 0, 0, max_len - pixel_values_videos[i].shape[1]),
            )
        pixel_values_videos = torch.cat(pixel_values_videos, dim=0)

        video_grid_thw = torch.cat(video_grid_thw, dim=0)
        second_per_grid_ts = torch.cat(second_per_grid_ts, dim=0)
    if len(pixel_values_images) > 0:
        max_len = max([x.shape[0] for x in pixel_values_images])
        for i in range(len(pixel_values_images)):
            pixel_values_images[i] = pixel_values_images[i].unsqueeze(0)
            assert (
                pixel_values_images[i].ndim == 3
            ), f"pixel_values_images[i].ndim: {pixel_values_images[i].ndim}"
            pixel_values_images[i] = F.pad(
                pixel_values_images[i],
                (0, 0, 0, max_len - pixel_values_images[i].shape[1]),
            )
        pixel_values_images = torch.cat(pixel_values_images, dim=0)
        image_grid_thw = torch.cat(image_grid_thw, dim=0)
    batch = {
        "input_ids": input_ids,
        "label_ids": label_ids,
    }
    if len(pixel_values_videos) > 0:
        batch["pixel_values_videos"] = pixel_values_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["second_per_grid_ts"] = second_per_grid_ts
        batch["pixel_values_videos_lengths_per_sample"] = torch.tensor(
            pixel_values_videos_lengths_per_sample, dtype=torch.long
        ).view(-1, 1)
    if len(pixel_values_images) > 0:
        batch["pixel_values_images"] = pixel_values_images
        batch["image_grid_thw"] = image_grid_thw
        batch["pixel_values_images_lengths_per_sample"] = torch.tensor(
            pixel_values_images_lengths_per_sample, dtype=torch.long
        ).view(-1, 1)
    return batch


def construct_dataset(
    config: SFTDataConfig,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    hf_config: AutoConfig,
    max_length: int,
):
    dataset = util.load_data_from_disk_or_hf(
        config.dataset_name, config.dataset_subset, config.dataset_revision or None
    )
    dataset_list = []
    for split_name in config.dataset_train_split:
        logger.info(
            f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
        )
        dataset_list.append(dataset[split_name])
    train_dataset = concatenate_datasets(dataset_list)
    logger.info(f"Final dataset size = {len(train_dataset)}")
    try:
        test_dataset = dataset[config.dataset_test_split]
        if len(test_dataset) == 0:
            raise ValueError("Test dataset is empty")
    except Exception:
        train_test_split = train_dataset.train_test_split(
            test_size=config.dataset_test_size, shuffle=False
        )
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

    train_sft_dataset = SFTDataset(
        config,
        tokenizer=tokenizer,
        processor=processor,
        hf_config=hf_config,
        max_length=max_length,
        dataset=train_dataset,
    )
    test_sft_dataset = SFTDataset(
        config,
        tokenizer=tokenizer,
        processor=processor,
        hf_config=hf_config,
        max_length=max_length,
        dataset=test_dataset,
    )

    return train_sft_dataset, test_sft_dataset


class SFTDataset(Dataset):
    def __init__(
        self,
        config: SFTDataConfig,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        hf_config: AutoConfig,
        max_length: int,
        dataset,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.column_name = config.conversation_column_name
        self.dataset = dataset
        self.max_length = max_length

        self.cache = None
        if self.config.enable_dataset_cache:
            if self.enable_dataset_preprocess:
                logger.warning(
                    "it is no need to cache the preprocessed dataset, please set enable_dataset_cache = False"
                )
            else:
                # TODO(zjx): can we reuse the cachebetween different training jobs?
                # It's not stable yet, we only checked if the config is the same
                # If there are any problems, it is recommended that the user clears the cache folder
                cache_folder = os.path.join(
                    os.environ.get(
                        "COSMOS_CACHE",
                        os.path.join(os.path.expanduser("~"), ".cache/cosmos/"),
                    ),
                    "datasets_cache",
                    f"{self.config.dataset_name}-{config_hash(config)}",
                )
                logger.info(f"SFTDataset Cache folder: {cache_folder}")
                self.cache = cache.DiskCache(cache_folder)
        self.vision_enabled = (
            self.config.vision_asset_column_name is not None
            and len(self.config.vision_asset_column_name) > 0
        )
        if self.vision_enabled and not self.config.enable_dataset_preprocess:
            cache_dir = os.environ.get(
                "COSMOS_CACHE", os.path.join(os.path.expanduser("~"), ".cache/cosmos/")
            )
            video_clips_path = os.path.join(
                cache_dir,
                "datasets",
                basename_from_modelpath(self.config.dataset_name),
                self.config.dataset_subset,
                "video_clips",
            )
            if not os.path.exists(video_clips_path):
                raise FileNotFoundError(
                    f"Dataset directory {video_clips_path} does not exist. Please check the dataset path."
                )
            mm_files_paths = {}
            for root, dirs, files in os.walk(video_clips_path):
                for file in files:
                    if file.endswith(
                        (".mp4", ".avi", ".mov")
                    ):  # Common video extensions
                        mm_files_paths[file] = os.path.join(root, file)
            self.mm_files_paths = mm_files_paths
        if hasattr(hf_config, "image_token_id"):
            self.image_token = tokenizer.decode([hf_config.image_token_id])
            self.image_token_id = hf_config.image_token_id
        else:
            self.image_token = None
            self.image_token_id = None
        if hasattr(hf_config, "video_token_id"):
            self.video_token = tokenizer.decode([hf_config.video_token_id])
            self.video_token_id = hf_config.video_token_id
        else:
            self.video_token = None
            self.video_token_id = None

    def _get_item_on_the_fly_(self, idx):
        item = self.dataset[idx]
        conversations = copy.deepcopy(item[self.column_name])

        if self.vision_enabled:
            video_path = self.mm_files_paths[item["video"].split("/")[-1]]
            multi_modal_content = {
                "type": "video",
                "video": video_path,
                "max_pixels": self.config.max_pixels,
                "fps": self.config.fps,
            }
            for conv in conversations:
                if conv["role"] == "user":
                    assert isinstance(
                        conv["content"], str
                    ), "User message must be string"
                    # Rewrite to support image/video tokens
                    content = []
                    content.append(multi_modal_content)
                    content.append(
                        {
                            "type": "text",
                            "text": conv["content"],
                        }
                    )
                    conv["content"] = content
            prompt = self.processor.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
            )
            image_inputs, video_inputs = process_vision_info(conversations)
            inputs = self.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            result_dict = {
                "input_ids": inputs["input_ids"][0].tolist()[: self.max_length],
                "pixel_values_videos": inputs["pixel_values_videos"],
                "video_grid_thw": inputs["video_grid_thw"],
                "second_per_grid_ts": torch.tensor(
                    inputs["second_per_grid_ts"], dtype=torch.float
                ),
                "pixel_values_videos_lengths_per_sample": inputs[
                    "pixel_values_videos"
                ].shape[0],
            }
        else:
            token_ids = self.tokenizer.apply_chat_template(
                conversations,
                return_assistant_tokens_mask=True,
                return_dict=True,
                add_generation_prompt=False,
            )["input_ids"]

            x = token_ids[: self.max_length]
            result_dict = {"input_ids": x}

        return result_dict

    def _get_item_preprocessed_(self, idx):
        item = self.dataset[idx]
        conversations = copy.deepcopy(item[self.column_name])

        assets_dicts = []
        if self.vision_enabled:
            assets = self.dataset[idx][self.config.vision_asset_column_name]
            cache_dir = os.environ.get(
                "COSMOS_CACHE", os.path.join(os.path.expanduser("~"), ".cache/cosmos/")
            )
            video_tensors_path = os.path.join(
                cache_dir,
                "datasets",
                self.config.dataset_name,
                self.config.dataset_subset,
                "video_tensors",
                f"fps-{self.config.fps}-pixels-{self.config.max_pixels}",
            )
            assert os.path.exists(
                video_tensors_path
            ), f"Dataset directory {video_tensors_path} does not exist. Please check the dataset path."
            if isinstance(assets, str):
                assets = [assets]
            else:
                assert isinstance(
                    assets, list
                ), "`vision_asset_column_name` must be a string or a list of strings"

            for asset in assets:
                asset = os.path.basename(asset).split(".")[0]
                asset_path = os.path.join(video_tensors_path, f"{asset}.cosmos")
                assert os.path.exists(asset_path), f"Asset {asset_path} does not exist"
                loaded_asset = torch.load(asset_path, map_location="cpu")
                assets_dicts.append(loaded_asset)
        # Check if the image/video token is already in the conversation
        # If so, we don't need to add the image/video token to the conversation
        if not self.vision_enabled:
            token_ids = self.tokenizer.apply_chat_template(
                conversations,
                return_assistant_tokens_mask=True,
                return_dict=True,
                add_generation_prompt=False,
            )["input_ids"]

            x = token_ids[: self.max_length]
            return {"input_ids": x}
        else:
            # Add image/video tokens to the conversation
            # text = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)

            added_placeholders = []
            for conv in conversations:
                # Hack the first user message to support image/video tokens
                if conv["role"] == "user":
                    assert isinstance(
                        conv["content"], str
                    ), "User message must be string"
                    # Rewrite to support image/video tokens
                    content = []
                    for x in assets_dicts:
                        placeholder = (
                            "video"
                            if x.get("pixel_values_videos") is not None
                            else "image"
                        )
                        content.append(
                            {
                                "type": placeholder,
                                "video": "",
                            }
                        )
                        n_tokens = x[f"n_{placeholder}_tokens"]
                        added_placeholders.append((placeholder, n_tokens))

                    content.append(
                        {
                            "type": "text",
                            "text": conv["content"],
                        }
                    )
                    conv["content"] = content
                    break

            text = self.processor.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=False
            )
            temp_image_token = "[COSMOS_IMAGE_TOKEN]"
            temp_video_token = "[COSMOS_VIDEO_TOKEN]"
            for placeholder, n_tokens in added_placeholders:
                if placeholder == "image":
                    text = text.replace(
                        self.image_token, n_tokens * temp_image_token, 1
                    )
                elif placeholder == "video":
                    text = text.replace(
                        self.video_token, n_tokens * temp_video_token, 1
                    )

            text = text.replace(temp_image_token, self.image_token)
            text = text.replace(temp_video_token, self.video_token)

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            # video:
            #   pixel_values_videos
            #   video_grid_thw
            #   second_per_grid_ts
            # image:
            #   pixel_values_images
            #   image_grid_thw

            # Concatenate all the assets
            n_videos = sum([1 for x in added_placeholders if x[0] == "video"])
            n_images = sum([1 for x in added_placeholders if x[0] == "image"])

            result_dict = {"input_ids": tokens[: self.max_length]}

            if n_videos > 0:
                pixel_values_videos = []
                video_grid_thw = []
                second_per_grid_ts = []
                for x in assets_dicts:
                    pixel_values_videos.append(x["pixel_values_videos"])
                    video_grid_thw.append(x["video_grid_thw"])
                    second_per_grid_ts.append(
                        torch.tensor(x["second_per_grid_ts"], dtype=torch.float)
                    )
                result_dict["pixel_values_videos"] = torch.cat(
                    pixel_values_videos, dim=0
                )
                result_dict["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)
                result_dict["second_per_grid_ts"] = torch.cat(second_per_grid_ts, dim=0)
                result_dict["pixel_values_videos_lengths_per_sample"] = result_dict[
                    "pixel_values_videos"
                ].shape[0]
            if n_images > 0:
                pixel_values_images = []
                image_grid_thw = []
                for x in assets_dicts:
                    pixel_values_images.append(x["pixel_values_images"])
                    image_grid_thw.append(x["image_grid_thw"])
                result_dict["pixel_values_images"] = torch.cat(
                    pixel_values_images, dim=0
                )
                result_dict["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
                result_dict["pixel_values_images_lengths_per_sample"] = result_dict[
                    "pixel_values_images"
                ].shape[0]

            return result_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.config.enable_dataset_preprocess:
            item = self._get_item_preprocessed_(idx)
        else:
            # we only cache on_the_fly result
            if self.cache is not None:
                cache_obj = self.cache.get(idx)
                if cache_obj is not None:
                    return cache_obj

            item = self._get_item_on_the_fly_(idx)

            if self.cache is not None:
                # try cache obj
                self.cache.set(idx, item)
        return item


class SFTTrainer(Trainer):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims):
        super(SFTTrainer, self).__init__(config, parallel_dims)
        if config.train.resume:
            try:
                self.ckpt_manager.load_checkpoint(
                    model=self.model,
                    optimizer=self.optimizers,
                    scheduler=self.lr_schedulers,
                )
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
            processor=self.hf_processor,
            hf_config=self.hf_config,
            max_length=config.policy.model_max_length,
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
            num_workers=config.train.train_policy.dataloader_num_workers,
            prefetch_factor=config.train.train_policy.dataloader_prefetch_factor,
            sampler=train_sampler,
            collate_fn=functools.partial(
                collate_fn,
                pad_token_id=self.tokenizer.pad_token_id,
                seq_len_multiple=self.seq_len_multiple,
                vision_ids=[train_dataset.image_token_id, train_dataset.video_token_id],
                # TODO(cjx): PP only support fixed length training data, fix it.
                fixed_length=config.policy.model_max_length
                if parallel_dims.pp_enabled
                else None,
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
                vision_ids=[val_dataset.image_token_id, val_dataset.video_token_id],
                # TODO(cjx): PP only support fixed length training data, fix it.
                fixed_length=config.policy.model_max_length
                if parallel_dims.pp_enabled
                else None,
            ),
        )
        # For iteration control
        self.total_steps = (
            len(self.train_data_loader) * config.train.epoch // self.dp_world_size
        )
        self.train_step = 0

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def validate(self):
        logger.info(
            f"[Policy] Validation at step {self.train_step}/{self.total_steps}..."
        )
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
                val_position_ids, val_pos_seq_dim = self.model.get_position_ids(
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
                                **val_batch, position_ids=val_position_ids
                            )
                        else:
                            pp_out = self.pp_scheduler_val.step(
                                position_ids=val_position_ids
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
            logger.info(f"[Policy] Validation loss: {val_avg_loss}")
        self.model.train()
        return val_avg_loss

    def train(self):
        self.profiler.start()
        pp_last_stage = False

        for batch in self.train_data_loader:
            start_time = time.time()
            for k, v in batch.items():
                batch[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v

            inputs = batch["input_ids"]
            labels = batch.pop("label_ids")

            position_ids, pos_seq_dim = self.model.get_position_ids(**batch)

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
                    targets, losses = (labels, []) if pp_last_stage else (None, None)
                    if pp_first_stage:
                        self.pp_scheduler.step(**batch, position_ids=position_ids)
                    else:
                        # FWD + BWD if it is 1F1B-like scheduler
                        self.pp_scheduler.step(
                            position_ids=position_ids, target=targets, losses=losses
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
            end_time = time.time()

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

            step_logging = True
            if self.config.logging.enable_logging:
                step_logging = is_wandb_available()
                if self.global_rank == 0:
                    iter_time = end_time - start_time
                    report_data = {
                        "train/loss_avg": global_avg_loss,
                        "train/loss_max": global_max_loss,
                        "train/learning_rate": self.lr_schedulers.get_last_lr()[0],
                        "train/iteration_time": iter_time,
                    }

                    # FIXME(dinghaoy): only compute MFU of rank 0, if enable tp or pp,
                    # it will be inaccurate. Need a reduce for all the metrics.
                    if self.config.logging.report_mfu:
                        mfu = compute_mfu(
                            model=self.model,
                            inputs=batch,
                            iter_time=iter_time,
                            num_gpus=self.world_size,
                            dtype=self.config.train.param_dtype,
                        )
                        for k, v in mfu.items():
                            report_data[f"train/{k}"] = v
                    if is_wandb_available():
                        log_wandb(
                            data=report_data,
                            step=self.train_step,
                        )
                    else:
                        logger.info(
                            f"Step: {self.train_step}/{self.total_steps}, Loss: {global_avg_loss:.5f}, Learning rate: {self.lr_schedulers.get_last_lr()[0]:.5e}, Iteration time: {iter_time:.3f}s."
                        )

            if step_logging:
                logger.info(
                    f"Step: {self.train_step}/{self.total_steps}, Loss: {global_avg_loss:.5e}"
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
                        f"[Policy] Saving huggingface checkpoint at step {self.train_step} to {self.config.train.output_dir}..."
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
                    f"[Policy] Saving cosmos checkpoint at step {self.train_step}..."
                )
                self.ckpt_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizers,
                    scheduler=self.lr_schedulers,
                    step=self.train_step,
                )
                self.save_manager.save_check(
                    step=self.train_step,
                    val_score=val_score,
                    pp_enabled=self.parallel_dims.pp_enabled,
                    pp_last_stage=pp_last_stage,
                )

        # process the final step
        val_score = self.validate()
        if self.config.train.ckpt.export_safetensors:
            logger.info(
                f"[Policy] Saving final huggingface checkpoint to {self.config.train.output_dir}..."
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

        logger.info(
            f"[Policy] Training finished at step {self.train_step}/{self.total_steps}, saving final cosmos checkpoint..."
        )
        self.ckpt_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizers,
            scheduler=self.lr_schedulers,
            step=self.train_step,
            is_final=True,
        )
        self.save_manager.save_check(
            step=self.train_step,
            val_score=val_score,
            pp_enabled=self.parallel_dims.pp_enabled,
            pp_last_stage=pp_last_stage,
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
