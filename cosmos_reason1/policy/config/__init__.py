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

from dataclasses import dataclass, field, is_dataclass, asdict, MISSING
from datetime import datetime
from typing import Any, Union, Optional, List
import os
import json
import hashlib
import torch
from cosmos_reason1.utils.util import update_dataclass_with_dict
from cosmos_reason1.utils.modelscope import update_config_if_modelscope


def skip_ui_field(*, default=MISSING, default_factory=MISSING, **kwargs):
    metadata = kwargs.pop("metadata", {})
    metadata["skip_ui"] = True
    if default_factory is not MISSING:
        return field(default_factory=default_factory, metadata=metadata, **kwargs)
    elif default is not MISSING:
        return field(default=default, metadata=metadata, **kwargs)
    else:
        raise ValueError("Must provide either default or default_factory.")


def config_hash(config) -> str:
    """
    Compute the hash of a config object
    """
    if is_dataclass(config):
        return hashlib.md5(json.dumps(asdict(config)).encode()).hexdigest()
    else:
        return "unhashable"


@dataclass
class SFTDataConfig:
    type: str = skip_ui_field(default="sft")

    dataset_name: str = field(
        default="",
        metadata={"help": "Huggingface dataset name or local path to parquet file"},
    )
    dataset_subset: Optional[str] = field(
        default="",
        metadata={"help": "Dataset subset if exists"},
    )
    dataset_revision: Optional[str] = field(
        default="",
        metadata={
            "help": "Dataset git revision if exist, can be a branch name, a tag, or a commit hash."
        },
    )
    dataset_train_split: Union[str, List[str]] = field(
        default_factory=list,
        metadata={"help": "A list of dataset splits to train"},
    )
    dataset_test_split: str = field(
        default="test", metadata={"help": "Dataset split to test"}
    )
    dataset_test_size: Union[float, int] = field(
        default=0.1,
        metadata={
            "help": "Size of the test set. If float, it is the ratio (between 0.0 and 1.0) of the dataset; if int, it is the absolute size of the test set."
        },
    )
    dataloader_shuffle: bool = field(
        default=False,
        metadata={
            "help": "Shuffle the dataloader. If False, the dataloader will be used in the order it is loaded."
        },
    )
    enable_dataset_cache: bool = field(
        default=False,
        metadata={
            "help": "Enable dataset cache process results, maybe accelerate the dataset loading",
        },
    )
    dataloader_num_workers: int = field(
        default=0, metadata={"help": "Number of subprocess to use for data loading"}
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of batches loaded in advance by each worker.",
        },
    )
    conversation_column_name: str = field(
        default="conversations",  # "conversation",
        metadata={"help": "Column name for formated conversation json"},
    )
    system_prompt: str = field(
        default="",
        metadata={
            "help": "System prompt for the model, which will be prepended to the prompt",
        },
    )

    def __post_init__(self):
        if self.dataloader_num_workers <= 0:
            self.dataloader_prefetch_factor = None
            self.dataloader_num_workers = 0


@dataclass
class CheckpointConfig:
    enable_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Enable checkpointing for training. If set to False, no checkpoint will be saved."
        },
    )

    save_freq: int = field(
        default=20, metadata={"help": "Checkpoint save frequency for training steps"}
    )
    save_mode: str = field(
        default="async",
        metadata={
            "choices": ["async", "sync"],
            "help": "Checkpoint save mode for training steps",
        },
    )
    max_keep: int = field(
        default=5,
        metadata={
            "help": "Maximum number of checkpoints to keep. If set to -1, all checkpoints will be kept."
        },
    )
    export_safetensors: bool = field(
        default=True,
        metadata={
            "help": "Whether to export a safetensors weight for huggingface usage, include related config files."
        },
    )
    upload_hf: bool = field(
        default=False,
        metadata={"help": "Whather to upload the safetensors weight to huggingface."},
    )
    hf_repo_name: str = field(
        default="Comos-Reason1",
        metadata={
            "help": "The huggingface repo name to upload the safetensors weight."
        },
    )
    upload_s3: Union[bool, str] = field(
        default=False,
        metadata={
            "help": "Whether to upload the checkpoint and safetensors to S3. Default to False, set `final` will upload the final checkpoint, `all` will upload all checkpoints."
        },
    )
    s3_bucket: str = field(
        default="cosmos-reason1",
        metadata={
            "help": "The S3 bucket name to upload the checkpoint and safetensors weight."
        },
    )
    s3_prefix: str = field(
        default="outputs",
        metadata={
            "help": "The S3 prefix to upload the checkpoint and safetensors weight."
        },
    )

    def __post_init__(self):
        if self.save_mode not in ["async", "sync"]:
            raise ValueError(
                f"Invalid save_mode: {self.save_mode}. Must be one of ['async', 'sync']"
            )
        if self.save_freq <= 0:
            raise ValueError(f"save_freq must be greater than 0, got {self.save_freq}")


@dataclass
class OverlongRewardConfig:
    enable_overlong_penalty: bool = field(
        default=False,
        metadata={
            "help": "Enable overlong penalty for the model. If set to True, the output will be penalized for responses that are too long."
        },
    )
    buffer_length: int = field(
        default=4096,
        metadata={
            "help": "Length of the buffer for overlong penalty. If the response length exceeds this value, the output will be penalized."
        },
    )
    penalty_factor: float = field(
        default=1.0,
        metadata={
            "help": "Penalty factor for overlong penalty. The penalry increases linearly with the length of the response exceeding the buffer length from 0 to the penalty_factor."
        },
    )


@dataclass
class GrpoConfig:
    type: str = skip_ui_field(default="grpo")
    variant: str = field(
        default="grpo",
        metadata={
            "help": "Variant of the GRPO, currently support `grpo`, and `dapo`",
            "choices": ["grpo", "dapo"],
        },
    )
    dataset_name: str = field(
        default="",
        metadata={"help": "Huggingface dataset name or local path to parquet file"},
    )
    dataset_subset: Optional[str] = field(
        default="",
        metadata={"help": "Dataset subset if exists"},
    )
    dataset_revision: Optional[str] = field(
        default="",
        metadata={
            "help": "Dataset git revision if exist, can be a branch name, a tag, or a commit hash."
        },
    )
    dataset_train_split: Union[str, List[str]] = field(
        default_factory=list,
        metadata={"help": "A list of dataset splits to train"},
    )
    dataloader_shuffle: bool = field(
        default=True,
        metadata={
            "help": "Shuffle the dataloader. If False, the dataloader will be used in the order it is loaded."
        },
    )
    enable_dataset_cache: bool = field(
        default=False,
        metadata={
            "help": "Enable dataset cache process results, maybe accelerate the dataset loading",
        },
    )
    dataloader_num_workers: int = field(
        default=0, metadata={"help": "Number of subprocess to use for data loading"}
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of batches loaded in advance by each worker.",
        },
    )
    prompt_column_name: str = field(
        default="",
        metadata={"help": "Column name for prompt"},
    )
    response_column_name: str = field(
        default="",
        metadata={"help": "Column name for response/reference answer"},
    )
    reward_function: List[str] = field(
        default_factory=list,
        metadata={
            "help": "A List of reward functions for the model. Currently support `single_choice`, `boxed_math`, and `format`. ",
        },
    )
    temperature: float = field(
        default=0.9,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )

    epsilon_low: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )

    epsilon_high: float = field(
        default=0.2,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )

    overlong_reward: OverlongRewardConfig = field(
        default_factory=OverlongRewardConfig,
        metadata={
            "help": "Configuration for overlong reward penalty. If enabled, the output will be penalized for responses that are too long."
        },
    )

    kl_beta: float = field(
        default=0.0,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )

    mu_iterations: int = field(
        default=1,
        metadata={
            "help": "Number of iterations per batch (denoted as μ in the algorithm)."
        },
    )
    mini_batch: int = field(
        default=2,
        metadata={"help": "mini-batch size for GRPO training."},
    )

    allowed_outdated_steps: int = field(
        default=10,
        metadata={
            "help": "Allowed outdated-async steps for rollout engine. "
            "If the number of left pending rollouts is larger than the `allowed_outdated_steps * n_policy_replicas * train_batch_per_replica`, "
            "then rollout engine traffic will be throttled. "
        },
    )

    def __post_init__(self):
        assert self.variant in [
            "grpo",
            "dapo",
        ], "variant must be one of ['grpo', 'dapo']"
        if self.dataloader_num_workers <= 0:
            self.dataloader_prefetch_factor = None
            self.dataloader_num_workers = 0


@dataclass
class SubProfilerConfig:
    do_profile: bool = field(
        default=False, metadata={"help": "Whether to profile, only used in runtime."}
    )
    active_steps: int = field(default=1, metadata={"help": "Number of active steps"})
    rank_filter: List[int] = field(
        default_factory=list, metadata={"help": "Rank filter"}
    )
    record_shape: bool = field(
        default=False, metadata={"help": "Whether to record shape"}
    )
    profile_memory: bool = field(
        default=False, metadata={"help": "Whether to profile memory"}
    )
    with_stack: bool = field(
        default=False, metadata={"help": "Whether to profile stack"}
    )
    with_modules: bool = field(
        default=False, metadata={"help": "Whether to profile modules"}
    )


@dataclass
class ProfilerConfig:
    enable_profiler: bool = field(
        default=False,
        metadata={
            "help": "Enable profiler for training",
        },
    )

    sub_profiler_config: SubProfilerConfig = field(
        default_factory=SubProfilerConfig, metadata={"help": "Sub profiler config"}
    )


@dataclass
class TrainingConfig:
    train_policy: Union[SFTDataConfig, GrpoConfig] = field(
        default_factory=SFTDataConfig
    )
    ckpt: CheckpointConfig = field(default_factory=CheckpointConfig)
    resume: Union[bool, str] = field(
        default=False,
        metadata={
            "help": "Resume training from a checkpoint. If True, will resume from the latest checkpoint of the `output_dir`. If a string, will resume from the specified checkpoint path."
        },
    )
    epoch: int = field(default=1, metadata={"help": "Number of epochs for training"})
    output_dir: str = field(default="./outputs", metadata={"help": "Output directory"})
    timestamp: str = skip_ui_field(
        default="",
        metadata={
            "help": "Timestamp for the output directory and wandb ID, if not set, will be generated automatically"
        },
    )
    epsilon: float = field(default=1e-6, metadata={"help": "Epsilon for optimizer"})
    optm_name: str = field(
        default="AdamW",
        metadata={"choices": ["AdamW", "Adam"], "help": "Optimizer name"},
    )
    optm_lr: float = field(
        default=1e-6, metadata={"help": "Learning rate for optimizer"}
    )
    optm_impl: str = field(
        default="fused",
        metadata={
            "choices": ["fused", "foreach", "for-loop"],
            "help": "More info: https://pytorch.org/docs/stable/optim.html",
        },
    )
    optm_weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for optimizer"}
    )
    optm_betas: tuple[float, float] = field(
        default=(0.9, 0.999), metadata={"help": "Betas for optimizer"}
    )
    optm_warmup_steps: int = field(
        default=20, metadata={"help": "Warmup steps for optimizer"}
    )
    optm_grad_norm_clip: float = field(
        default=1.0, metadata={"help": "Gradient norm clip for optimizer"}
    )

    async_tp_enabled: bool = field(
        default=False, metadata={"help": "Whether to use async tensor parallelism"}
    )

    compile: bool = field(
        default=True, metadata={"help": "Whether to use torch.compile"}
    )

    param_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": "The data type for parameters and activations",
            "choices": ["bfloat16", "float16", "float32"],
        },
    )

    fsdp_reduce_dtype: str = field(
        default="float32",
        metadata={
            "help": "The data type for reduction in FSDP",
            "choices": ["float32"],
        },
    )
    fsdp_offload: bool = field(
        default=False,
        metadata={"help": "Whether to offload the model to CPU if using FSDP"},
    )

    fsdp_reshard_after_forward: str = field(
        default="default",
        metadata={
            "help": "Reshard the param after forward pass in FSDP",
            "choices": ["always", "never", "default"],
        },
    )

    train_batch_per_replica: int = field(
        default=8,
        metadata={
            "help": "The batch size for training per iteration in one replica, this is the local batch size for each gradient accumulation step",
        },
    )

    enable_validation: bool = field(
        default=False,
        metadata={"help": "Enable validation during training."},
    )
    validation_freq: int = field(
        default=20,
        metadata={
            "help": "Validation frequency during training, in terms of training steps",
        },
    )
    validation_batch_per_replica: int = field(
        default=24,
        metadata={
            "help": "The batch size for validation per iteration in one replica.",
        },
    )

    sync_weight_interval: int = field(
        default=1,
        metadata={
            "help": "The interval of train step for synchronizing weights between replicas. "
        },
    )

    def __post_init__(self):
        self.ckpt.__post_init__()
        if self.async_tp_enabled and not self.compile:
            raise ValueError(
                "Async tensor parallelism requires torch.compile to be enabled"
            )

    def key_values(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @property
    def param_torch_dtype(self):
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.param_dtype]

    @property
    def fsdp_reduce_torch_dtype(self):
        return {"float32": torch.float32}[self.fsdp_reduce_dtype]


@dataclass
class ParallelismConfig:
    n_init_replicas: int = field(
        default=1, metadata={"help": "Number of initial replicas to be created"}
    )
    tp_size: int = field(default=2, metadata={"help": "Tensor parallelism size"})
    cp_size: int = field(default=1, metadata={"help": "Context parallelism size"})
    dp_shard_size: int = field(
        default=-1, metadata={"help": "Data Parallelism size in sharded mode"}
    )
    pp_size: int = field(default=1, metadata={"help": "Pipeline parallelism size"})
    pp_dynamic_shape: bool = field(
        default=True, metadata={"help": "Pipeline parallelism dynamic shape"}
    )
    pp_micro_batch_size: int = field(
        default=1,
        metadata={
            "help": "Pipeline parallelism micro batch size, `n_micro_batch = batch_size / pp_micro_batch_size`, which must be divisible by `pp` stages"
        },
    )
    dp_replicate_size: int = skip_ui_field(
        default=1,
        metadata={
            "help": "Data Parallelism size in replica mode. Only configurable in SFT type job, must be 1 in GRPO type job for dynamic scaling support purpose.",
            "choices": [1],
        },
    )

    cp_rotate_method: str = field(
        default="allgather",
        metadata={
            "help": "The method to rotate kv shards during context parallelism",
            "choices": ["allgather", "alltoall"],
        },
    )

    @property
    def world_size(self):
        world_size = os.environ.get("WORLD_SIZE", 1)
        return int(world_size)

    @property
    def local_world_size(self):
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE", 1)
        return int(local_world_size)

    def key_values(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class PolicyConfig:
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    model_name_or_path: str = field(
        # default="Qwen/Qwen2.5-3B-Instruct",  #'Qwen/Qwen2.5-VL-7B-Instruct'
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        metadata={
            "help": "The model name or path, compatible with huggingface model name or local path"
        },
    )
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "The maximum length for training, longer than this will be ignored for training stability"
        },
    )
    model_gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Whether to use gradient checkpointing"}
    )

    def key_values(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class RolloutParallelismConfig(ParallelismConfig):
    n_init_replicas: int = field(
        default=1, metadata={"help": "Number of initial replicas to be created"}
    )
    tp_size: int = field(default=2, metadata={"help": "Tensor parallelism size"})
    pp_size: int = field(default=1, metadata={"help": "Pipeline parallelism size"})

    # Fields below are that we do not want user to config it.
    dp_replicate_size: int = skip_ui_field(
        default=1,
        metadata={
            "help": "Data Parallelism size in replica mode, only 1 is supported for dynamic scaling purpose.",
            "choices": [1],
        },
    )
    cp_size: int = skip_ui_field(
        default=1, metadata={"help": "Context parallelism size"}
    )
    dp_shard_size: int = skip_ui_field(
        default=-1, metadata={"help": "Data Parallelism size in sharded mode"}
    )
    cp_rotate_method: str = skip_ui_field(
        default="allgather",
        metadata={
            "help": "The method to rotate kv shards during context parallelism",
            "choices": ["allgather", "alltoall"],
        },
    )


@dataclass
class SamplingConfig:
    temperature: float = field(
        default=0.9, metadata={"help": "Temperature for sampling."}
    )
    top_p: float = field(default=1.0, metadata={"help": "Top-p for sampling."})
    top_k: int = field(default=10, metadata={"help": "Top-k for sampling."})
    repetition_penalty: float = field(
        default=1.0, metadata={"help": "Repetition penalty for sampling."}
    )
    use_flashinfer: bool = field(
        default=False, metadata={"help": "Use flashinfer for sampling."}
    )


@dataclass
class RolloutConfig:
    parallelism: RolloutParallelismConfig = field(
        default_factory=RolloutParallelismConfig
    )
    gpu_memory_utilization: float = field(
        default=0.8,
        metadata={"help": "GPU memory utilization factor for rollout backend."},
    )
    enable_chunked_prefill: bool = field(
        default=False, metadata={"help": "Whether to enable chunked prefill for vLLM."}
    )
    max_response_length: int = field(
        default=2048, metadata={"help": "Max output length of rollout generation."}
    )
    n_generation: int = field(
        default=16, metadata={"help": "n parameter same like what in OpenAI chat API."}
    )

    batch_size: int = skip_ui_field(
        default=1, metadata={"help": "Batch size for rollout."}
    )

    # not used yet.
    quantization: str = skip_ui_field(
        default="none",
        metadata={
            "help": "Quantization in vllm rollout generation.",
            "choices": ["none"],
        },
    )

    seed: int = field(default=42, metadata={"help": "random seed for rollout."})

    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    vllm_use_flashinfer: bool = field(
        default=False, metadata={"help": "Use flashinfer for vllm rollout."}
    )

    def __post_init__(self):
        if isinstance(self.parallelism, dict):
            self.parallelism = RolloutParallelismConfig(**self.parallelism)

    def key_values(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class LoggingConfig:
    enable_logging: bool = field(
        default=False,
        metadata={"help": "Enable wandb logging for training."},
    )
    project_name: str = field(
        default="cosmos_reason1",
        metadata={
            "help": "Wandb project name for logging. If set, the training will be logged to this project."
        },
    )
    experiment_name: str = field(
        default=None,
        metadata={
            "help": "A short display name for this run. If not set, will use the `output_dir` as the experiment name.",
        },
    )
    report_mfu: bool = field(
        default=False,
        metadata={
            "help": "Whether to report the MFU (Model FLOPs Utilization) to wandb."
        },
    )


@dataclass
class Config:
    train: TrainingConfig = field(default_factory=TrainingConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    redis: str = skip_ui_field(
        default="",
        metadata={
            "help": "Redis server address port, format: port",
        },
    )
    eth_ips: str = skip_ui_field(
        default="",
        metadata={
            "help": "List of eth ip addresses, format: ip1;ip2;ip3",
        },
    )

    def key_values(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, config_data: dict[str, Any]) -> "Config":
        config = Config()

        if "train" in config_data:
            # Set unique timestamp for output directory
            if (
                "timestamp" not in config_data["train"]
                or config_data["train"]["timestamp"] == ""
            ):
                config_data["train"]["timestamp"] = datetime.now().strftime(
                    "%Y%m%d%H%M%S"
                )
                config_data["train"]["output_dir"] = os.path.join(
                    config_data["train"]["output_dir"],
                    config_data["train"]["timestamp"],
                )

            # Handle train_policy type before general update
            if "train_policy" in config_data["train"]:
                train_policy_data = config_data["train"]["train_policy"]

                # Determine the type based on characteristic fields
                if any(
                    key in train_policy_data
                    for key in ["temperature", "epsilon_low", "epsilon_high", "kl_beta"]
                ):
                    config.train.train_policy = GrpoConfig()
                else:
                    config.train.train_policy = SFTDataConfig()

        update_dataclass_with_dict(config, config_data)
        config.validate()
        config = update_config_if_modelscope(config)
        return config

    def validate(self):
        assert (
            self.policy.model_name_or_path is not None
            and self.policy.model_name_or_path != ""
        ), "model_name_or_path is required"
        assert self.policy.parallelism.tp_size > 0, "tp_size must be greater than 0"
        assert self.policy.parallelism.cp_size > 0, "cp_size must be greater than 0"
        assert self.policy.parallelism.pp_size > 0, "pp_size must be greater than 0"
        assert (
            self.policy.parallelism.dp_shard_size >= -1
            and self.policy.parallelism.dp_shard_size != 0
        ), "dp_shard_size must be greater than 0 or -1 to be auto-inferred"
        if self.policy.parallelism.pp_size > 1:
            assert (
                self.policy.parallelism.pp_micro_batch_size > 0
            ), "pp_micro_batch_size must be greater than 0"
            assert (
                self.train.train_batch_per_replica
                % self.policy.parallelism.pp_micro_batch_size
                == 0
            ), "train_batch must be divisible by pp_micro_batch_size"

            # Here we assume that PP uses `Single-stage per rank` which is true for:
            #   - GPipe
            #   - 1F1B
            # But not correct for those `InterleavedXXX` style schedule
            assert (
                (
                    self.train.train_batch_per_replica
                    // self.policy.parallelism.pp_micro_batch_size
                )
                % self.policy.parallelism.pp_size
                == 0
            ), "train_batch / pp_micro_batch_size must be divisible by pp_size"
        if self.train.train_policy.type == "grpo":
            if isinstance(self.train.train_policy.reward_function, str):
                self.train.train_policy.reward_function = [
                    self.train.train_policy.reward_function
                ]
            assert (
                len(self.train.train_policy.reward_function) > 0
            ), "reward_function must be a list of reward functions"
        if isinstance(self.train.train_policy.dataset_train_split, str):
            self.train.train_policy.dataset_train_split = [
                self.train.train_policy.dataset_train_split
            ]
        if self.train.ckpt.upload_s3:
            if self.train.ckpt.upload_s3 not in ["final", "all"]:
                raise ValueError(
                    "upload_s3 must be one of ['final', 'all'] or False, got {}".format(
                        self.train.ckpt.upload_s3
                    )
                )
