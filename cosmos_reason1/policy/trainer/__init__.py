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

from cosmos_reason1.policy.model import build_model
from cosmos_reason1.policy.config import Config as CosmosConfig
from cosmos_reason1.utils.parallelism import (
    ParallelDims,
    train_context,
)
from cosmos_reason1.dispatcher.protocol import Role
import json
import os
import torch
import threading
from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.checkpoint import (
    upload_folder_to_s3,
    CheckpointMananger,
    SaveManager,
)
from transformers import AutoTokenizer, AutoConfig, AutoProcessor, GenerationConfig
from cosmos_reason1.policy.trainer.optm import build_optimizers, build_lr_schedulers
from cosmos_reason1.comm.base import CommMixin
from safetensors.torch import save_file
from huggingface_hub import create_repo, upload_folder, whoami
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from typing import Dict
import cosmos_reason1.utils.util as util
from cosmos_reason1.utils.profiler import CosmosProfiler


class Trainer(CommMixin):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims):
        super().__init__()
        self.config = config
        if self.config.policy.parallelism.dp_shard_size == -1:
            self.config.policy.parallelism.dp_shard_size = parallel_dims.dp_shard
        assert self.config.policy.parallelism.dp_shard_size == parallel_dims.dp_shard
        assert (
            self.config.policy.parallelism.dp_shard_size > 0
        ), "dp_shard_size must be greater than 0"
        self.parallel_dims = parallel_dims
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.role = Role.POLICY
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.init_comm()

        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        self.tokenizer = util.retry(AutoTokenizer.from_pretrained)(
            config.policy.model_name_or_path,
            trust_remote_code=True,
        )
        self.hf_config = util.retry(AutoConfig.from_pretrained)(
            config.policy.model_name_or_path,
            trust_remote_code=True,
        )
        try:
            self.hf_processor = util.retry(AutoProcessor.from_pretrained)(
                config.policy.model_name_or_path,
                trust_remote_code=True,
            )
        except Exception as e:
            self.hf_processor = None
            logger.info(
                f"Failed to load processor for {config.policy.model_name_or_path}, using tokenizer as processor, ignore if not needed, error = {e}"
            )

        self.train_stream = torch.cuda.current_stream()
        model = build_model(config)

        try:
            # Apply parallelism to the model
            parallelize_fn, _ = model.parallelize_fn
            self.pp_scheduler = parallelize_fn(
                model, parallel_dims, config, pp_loss_fn=self.pp_loss_fn
            )
            model.to_empty(device=self.device)
            model.post_to_empty_hook(config)
            torch.cuda.empty_cache()
            self.model_parts = model.separate_model_parts()
            self.model = model

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e

        self.ckpt_manager = CheckpointMananger(config, self.global_rank)
        # profiler is initialized after the init_comm()
        self.profiler = CosmosProfiler(
            config,
            parallel_dims,
            replica_name=self.replica_name,
            alternative_urls=self.get_alternative_urls("api/set_trace_path"),
        )
        self.save_manager = SaveManager(config, self.global_rank)
        # TODO(cjx): add `CompiledAutograd` support
        self.context = train_context(False)
        self.optimizers = build_optimizers(self.model_parts, self.config)
        self.lr_schedulers = build_lr_schedulers(self.optimizers, self.config)
        self.seq_len_multiple = parallel_dims.cp * parallel_dims.tp
        logger.info(
            f"Trainer initialized at local rank {self.local_rank}, with seq_len_multiple: {self.seq_len_multiple}"
        )

    @property
    def pp_loss_fn(self):
        raise NotImplementedError("pp_loss_fn must be provided by subclass")

    def export_safetensors(
        self,
        output_dir: str,
        rel_path: str,
        trainable_only: bool = False,
        is_final=False,
    ):
        path = os.path.join(output_dir, rel_path)
        if self.parallel_dims.dp_replicate_coord[0] > 0:
            return

        if self.global_rank == 0:
            logger.info(
                f"Prepare to exporting safetensors to {path} at rank {self.global_rank}"
            )
        torch.distributed.barrier()

        def get_tensor_size(tensor):
            """Get the size of the tensor in bytes."""
            return tensor.element_size() * tensor.numel()

        max_file_size_gb = 4
        max_size_bytes = max_file_size_gb * 1024**3  # 4 GB in bytes
        current_chunk = {}
        total_chunk_size = 0
        current_chunk_size = 0
        file_idx = 0
        manifest = {}  # Record the weight->file name mapping

        def create_file_name(pp_rank, pp_size, file_idx):
            if pp_size == 1:
                name = f"{file_idx:05d}.safetensors"
            else:
                name = f"model-{pp_rank}-of-{pp_size}-{file_idx:05d}.safetensors"
            return os.path.join(path, name)

        def save_chunked_tensors(
            chunk: Dict[str, torch.Tensor], chunk_size: int, file_path: str
        ):
            """
            Save a dictionary of tensors into a safetensors file.
            Only the rank 0 of dp_shard, cp, tp will save the file.
            """
            nonlocal total_chunk_size
            if (
                self.parallel_dims.dp_shard_coord[0] == 0
                and self.parallel_dims.cp_coord[0] == 0
                and self.parallel_dims.tp_coord[0] == 0
            ):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                for name, param in chunk.items():
                    manifest[name] = os.path.basename(file_path)
                total_chunk_size += chunk_size
                save_file(chunk, file_path)
                logger.info(f"Saved chunk {file_idx} to {os.path.basename(file_path)}")

        for name, param in self.model.to_hf_state_dict_generator(self.parallel_dims):
            if trainable_only and not param.requires_grad:
                continue
            is_dtensor = isinstance(param, torch.distributed.tensor.DTensor)
            param = param.full_tensor() if is_dtensor else param
            param = param.detach().data

            pp_rank, pp_size = self.parallel_dims.pp_coord

            for _name, _param in self.model.maybe_decompose_weights(name, param):
                tensor_size = get_tensor_size(_param)
                # If adding the current tensor exceeds the size limit, save the current chunk
                if current_chunk_size + tensor_size > max_size_bytes:
                    # Save the current chunk as a safetensor file
                    file_name = create_file_name(pp_rank, pp_size, file_idx)
                    save_chunked_tensors(current_chunk, current_chunk_size, file_name)

                    # Reset for the next chunk
                    current_chunk = {_name: _param}
                    current_chunk_size = tensor_size
                    file_idx += 1
                else:
                    # Add tensor to the current chunk
                    current_chunk[_name] = _param
                    current_chunk_size += tensor_size

        # Save any remaining tensors in the last chunk
        if current_chunk:
            file_name = create_file_name(pp_rank, pp_size, file_idx)
            save_chunked_tensors(current_chunk, current_chunk_size, file_name)

        # Allgather the manifest from all pipeline stages
        if self.parallel_dims.pp_enabled:
            pp_group = self.parallel_dims.mesh["pp"].get_group()
            pp_size = self.parallel_dims.pp
            output = [None for _ in range(pp_size)]
            tensor_sizes = [0 for _ in range(pp_size)]
            torch.distributed.all_gather_object(output, manifest, group=pp_group)
            torch.distributed.all_gather_object(
                tensor_sizes, total_chunk_size, group=pp_group
            )
            merged_manifest = {}
            for m in output:
                merged_manifest.update(m)
            total_tensor_size = sum(tensor_sizes)
        else:
            merged_manifest = manifest
            total_tensor_size = total_chunk_size
        if self.global_rank == 0:
            with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "total_size": total_tensor_size,
                        },
                        "weight_map": merged_manifest,
                    },
                    f,
                    indent=4,
                )
            # save hf_config and tokenizer_config
            self.hf_config.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            if self.hf_processor is not None:
                self.hf_processor.save_pretrained(path)
            # save the generation config to get the generation aligned with HF.
            try:
                generation_config = util.retry(GenerationConfig.from_pretrained)(
                    self.config.policy.model_name_or_path
                )
                generation_config.save_pretrained(path)
            except Exception:
                logger.warning("[Policy] No generation config found, do not save it.")
            # upload the final model to huggingface
            if self.config.train.ckpt.upload_hf and is_final:

                def upload_hf(folder_path, repo_id):
                    # hide redundant logs of huggingface
                    disable_progress_bars()
                    upload_folder(
                        folder_path=folder_path,
                        path_in_repo=".",
                        repo_id=repo_id,
                        commit_message="Upload model",
                    )
                    enable_progress_bars()
                    logger.info(f"Model uploaded to huggingface: {repo_id}")

                username = whoami()["name"]
                repo_id = (
                    username
                    + "/"
                    + self.config.train.ckpt.hf_repo_name
                    + "-"
                    + self.config.train.timestamp
                )
                create_repo(repo_id, exist_ok=True)
                logger.info(f"Uploading the final model to huggingface: {repo_id}...")
                upload_hf(path, repo_id)
            # upload the model to s3
            if self.config.train.ckpt.upload_s3:
                if is_final:
                    # syncronizely upload the final model to s3
                    upload_folder_to_s3(
                        path,
                        self.config.train.ckpt.s3_bucket,
                        os.path.join(self.config.train.ckpt.s3_prefix, rel_path),
                    )
                elif self.config.train.ckpt.upload_s3 == "all":
                    # asynchronously upload the model to s3
                    threading.Thread(
                        target=upload_folder_to_s3,
                        args=(
                            path,
                            self.config.train.ckpt.s3_bucket,
                            os.path.join(self.config.train.ckpt.s3_prefix, rel_path),
                        ),
                    ).start()

        torch.distributed.barrier()
        if self.global_rank == 0:
            logger.info(f"\n\nExported safetensors to {path}\n\n")
