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

import os
import re
import ast
import multiprocessing
import json
import datasets
import socket
import queue
import dataclasses
import base64
import struct
import tarfile
import ctypes
from functools import wraps
from msgpack import ExtType
from tqdm import tqdm
from typing import List
import torch
import pynvml
from contextlib import contextmanager
from torch.functional import F
from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    snapshot_download,
    HfFileSystem,
)
import time
import functools
from cosmos_reason1.utils.logging import logger
from safetensors import safe_open
from cosmos_reason1.utils.constant import CACHE_DIR


def create_cached_dir_if_needed():
    """
    Creates the local cached dir if it doesn't exist.
    """
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_data_from_disk_or_hf(data_name, data_subset=None, revision=None):
    if data_subset is not None and len(data_subset) == 0:
        data_subset = None

    if os.path.exists(data_name):
        try:
            return datasets.load_from_disk(data_name)
        except Exception as e:
            logger.warning(
                f"Failed to load dataset from disk: {e}. Trying to load from HuggingFace Hub..."
            )
            return datasets.load_dataset(data_name, data_subset, revision=revision)
    return datasets.load_dataset(data_name, data_subset, revision=revision)


def read_json_file(file_path):
    with open(file_path, "r") as file:
        json_data = json.load(file)
    return json_data


def write_json_file(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def resolve_model_path(model_path: str) -> str:
    if not os.path.exists(model_path.replace(":", "/")):
        if ":" in model_path:
            if model_path.count(":") > 1:
                raise ValueError(
                    f"Invalid model path {model_path}, should be in the format of 'owner/repo[:path]'"
                )
            model_path, file_path = model_path.split(":")
        else:
            model_path = model_path
            file_path = None

        # Check whether `model_path` is a HuggingFace repo id with repo name
        if len(model_path.split("/")) == 2:
            logger.info(
                f"model path {model_path} is not a directory. Trying to load from HuggingFace Hub..."
            )

            hf_fs = HfFileSystem(token=os.environ.get("HF_TOKEN", None))
            files = hf_fs.ls(model_path, detail=False)
            if (
                os.path.join(model_path, "model.safetensors.index.json") in files
                or os.path.join(model_path, "model.safetensors") in files
            ):
                logger.info(
                    f"Found safetensors in {model_path}. Ignoring *pytorch_model* and *consolidated* files."
                )
                ignore_patterns = ["*pytorch_model*", "*consolidated*"]
            else:
                ignore_patterns = None

            try:
                model_path = retry(snapshot_download)(
                    model_path,
                    token=os.environ.get("HF_TOKEN"),
                    cache_dir=os.environ.get(
                        "HF_HOME",
                        os.path.expanduser("~/.cache/huggingface/transformers/"),
                    ),
                    ignore_patterns=ignore_patterns,
                    allow_patterns=file_path,
                )
            except Exception as e:
                logger.error(f"Error: {e}")
                raise

            if file_path is not None:
                model_path = os.path.join(model_path, file_path)
            logger.info(f"Downloaded model from HuggingFace to {model_path}")

        else:
            raise ValueError(
                f"Model path {model_path} is not a directory and not a valid HuggingFace repo id with repo name."
            )
    else:
        model_path = model_path.replace(":", "/")
    return model_path


def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def find_available_port(start_port):
    max_port = 65535  # Maximum port number
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue

    raise RuntimeError("No available ports found in the specified range.")


def put_with_overwrite(q: queue.Queue, item):
    assert isinstance(q, queue.Queue), "q must be a queue.Queue, but not asyncio.Queue"
    if q.full():
        try:
            q.get_nowait()  # Remove the oldest item
        except queue.Empty:
            pass  # In case the queue becomes empty in a race condition
    q.put(item)


def extract_fields(dc_instance):
    """
    Recursively extract dataclass fields into a nested dictionary.
    Leaf nodes contain a dict with 'value', 'metadata', 'type', and 'input_type'.
    """

    def recursive_extract(instance):
        fields = {}
        for f in dataclasses.fields(instance):
            if f.metadata.get("skip_ui", False):
                continue
            value = getattr(instance, f.name)

            # Special handling for train_policy field
            if f.name == "train_policy" and hasattr(instance, "train_policy"):
                fields[f.name] = recursive_extract(
                    value
                )  # Just extract current policy's fields
                continue

            if dataclasses.is_dataclass(value):
                fields[f.name] = recursive_extract(value)
            else:
                field_data = {
                    "value": value,
                    "metadata": f.metadata,
                    "type": f.type,
                    "input_type": "checkbox" if f.type == bool else "text",
                }
                fields[f.name] = field_data
        return fields

    return recursive_extract(dc_instance)


def parse_collection(s):
    """
    Attempts to parse a string into a Python literal.

    If the string represents a list or tuple, it returns the corresponding
    Python object. Otherwise, it returns None.
    """
    try:
        result = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # The string is not a valid Python literal.
        return None

    # Check if the result is a list or tuple.
    if isinstance(result, (list, tuple)):
        return result
    else:
        return None


def update_dataclass(dc_instance, form, prefix=""):
    """
    Recursively update the dataclass instance based on form data.
    The form keys are built as nested keys
    """
    for f in dataclasses.fields(dc_instance):
        if f.metadata.get("skip_ui", False):
            continue

        field_name = f.name
        full_key = f"{prefix}.{field_name}" if prefix else field_name
        value = getattr(dc_instance, field_name)

        # Handle train_policy field specially
        if field_name == "train_policy" and hasattr(dc_instance, "train_policy"):
            # The train_policy instance is already set to the correct type (SFT or GRPO)
            # Just update its fields
            update_dataclass(value, form, prefix=full_key)
            continue

        if dataclasses.is_dataclass(value):
            update_dataclass(value, form, prefix=full_key)
        else:
            form_value = form.get(full_key)
            if form_value is None:
                if f.type == bool:
                    new_value = False
                else:
                    continue
            elif f.type == bool:
                new_value = form_value.lower() == "true"
            elif f.type == int:
                new_value = int(form_value)
            elif f.type == float:
                new_value = float(form_value)
            elif f.type == tuple:
                new_value = tuple(map(float, form_value.split(",")))
            else:
                parsed_set = parse_collection(form_value)
                new_value = parsed_set if parsed_set is not None else form_value
            setattr(dc_instance, field_name, new_value)


def update_dataclass_with_dict(dc_instance, config_data):
    if config_data is None:
        raise RuntimeError("Got null config.")
    for key, value in config_data.items():
        current_value = getattr(dc_instance, key)
        if dataclasses.is_dataclass(current_value):
            if value is None:
                continue
            update_dataclass_with_dict(current_value, value)
        else:
            setattr(dc_instance, key, value)


def list_to_b64(lst) -> str:
    # for int64_t listy to base64
    byte_data = struct.pack(f"{len(lst)}q", *lst)
    return base64.b64encode(byte_data).decode("utf-8")


def b64_to_list(b64_str) -> List[int]:
    # for base64 to int64_t list
    byte_data = base64.b64decode(b64_str)
    n = len(byte_data) // 8
    return list(struct.unpack(f"{n}q", byte_data))


@contextmanager
def cosmos_default_dtype(dtype: torch.dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


class IdentityLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = (
            selected_logits - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(
            logits, index
        ):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def clear_weight_name(name: str) -> str:
    return name.replace("._orig_mod", "").replace("._checkpoint_wrapped_module", "")


def _extract_tgz_file(tgz_file, output_dir):
    with tarfile.open(tgz_file, "r:gz") as tar:
        tar.extractall(path=output_dir)


def basename_from_modelpath(path: str) -> str:
    path = path.strip().strip("/")
    if len(path.split("/")) > 2:
        path = "/".join(path.split("/")[-2:])
    return path


def if_use_modelscope(path: str) -> bool:
    modelscope_cache_dir = os.environ.get(
        "MODELSCOPE_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache/modelscope_cache/dataset"),
    )
    return path.startswith(modelscope_cache_dir)


def prepare_cosmos_data(config):
    fps = config.train.train_policy.fps
    max_pixels = config.train.train_policy.max_pixels

    cache_dir = os.environ.get(
        "COSMOS_CACHE", os.path.join(os.path.expanduser("~"), ".cache/cosmos/")
    )
    dataset_name = basename_from_modelpath(config.train.train_policy.dataset_name)
    use_modelscope = if_use_modelscope(config.train.train_policy.dataset_name)
    dataset_dir = os.path.join(
        cache_dir,
        "datasets",
        dataset_name,
        config.train.train_policy.dataset_subset,
    )
    video_clips_dir = os.path.join(dataset_dir, "video_clips")
    video_tensors_dir = os.path.join(
        dataset_dir,
        "video_tensors",
        f"fps-{fps}-pixels-{max_pixels}",
    )

    if os.path.exists(video_clips_dir) and os.path.exists(video_tensors_dir):
        clip_files = []
        for root, dirs, files in os.walk(video_clips_dir):
            for file in files:
                if file.endswith((".mp4", ".avi", ".mov")):
                    clip_files.append(os.path.join(root, file))
        num_clips = len(clip_files)
        num_tensors = len(
            [
                tensor
                for tensor in os.listdir(video_tensors_dir)
                if tensor.endswith(".cosmos")
            ]
        )

        # dataset = load_data_from_disk_or_hf(
        #     config.train.train_policy.dataset_name,
        #     config.train.train_policy.dataset_subset,
        # )
        # num_samples = len(dataset[config.train.train_policy.dataset_train_split])
        if num_clips == num_tensors:
            logger.info(
                f"Dataset {config.train.train_policy.dataset_name} is already prepared."
            )
            return

    ## Prepare video clips
    re_pattern = re.compile(
        rf"^{re.escape(config.train.train_policy.dataset_subset)}/clips/.*\.tar\.gz$"
    )
    file_pattern = f"{config.train.train_policy.dataset_subset}/clips/*.tar.gz"
    if use_modelscope:
        assert os.path.exists(config.train.train_policy.dataset_name)
        # list all files in the local directory config.train.train_policy.dataset_name
        import glob

        remote_files = [
            f.replace(config.train.train_policy.dataset_name + "/", "")
            for f in glob.glob(
                f"{config.train.train_policy.dataset_name}/**/*.tar.gz", recursive=True
            )
        ]
    else:
        remote_files = list_repo_files(
            repo_id=dataset_name,
            repo_type="dataset",
            revision=config.train.train_policy.dataset_revision or None,
        )

    tgz_files = [f for f in remote_files if re_pattern.match(f)]
    if tgz_files:
        if use_modelscope:
            downloaded_clips_dir_path = os.path.join(
                config.train.train_policy.dataset_name,
                config.train.train_policy.dataset_subset,
                "clips",
            )
        else:
            downloaded_snapshot_directory_cache = retry(snapshot_download)(
                dataset_name,
                allow_patterns=[file_pattern],
                repo_type="dataset",
                revision=config.train.train_policy.dataset_revision or None,
            )

            downloaded_clips_dir_path = os.path.join(
                downloaded_snapshot_directory_cache,
                config.train.train_policy.dataset_subset,
                "clips",
            )
        assert os.path.exists(
            downloaded_clips_dir_path
        ), f"Can not find clips directory at {downloaded_clips_dir_path}"

        # Avoid redundant extraction
        if not os.path.exists(video_clips_dir):
            os.makedirs(video_clips_dir, exist_ok=True)
            with tqdm(
                total=len(tgz_files), desc="Extracting clips tar.gz files"
            ) as pbar:
                results = {}
                with multiprocessing.Pool(
                    processes=min(multiprocessing.cpu_count(), 8)
                ) as pool:
                    for tar_file in tgz_files:
                        full_file_path = os.path.join(
                            downloaded_clips_dir_path, os.path.basename(tar_file)
                        )
                        results[tar_file] = pool.apply_async(
                            _extract_tgz_file,
                            (full_file_path, video_clips_dir),
                            callback=lambda _: pbar.update(1),
                        )
                    pool.close()
                    pool.join()

                for tar_file, result in results.items():
                    if not result.successful():
                        raise RuntimeError(
                            f"Failed to extract {tar_file}: {result.get()}"
                        )

    else:
        # legacy dataset format with single tar gz file
        if not os.path.exists(video_clips_dir):
            os.makedirs(video_clips_dir, exist_ok=True)
            clip_filename = os.path.join(
                config.train.train_policy.dataset_subset, "clips.tar.gz"
            )
            if use_modelscope:
                clip_tgz = os.path.join(
                    config.train.train_policy.dataset_name, clip_filename
                )
            else:
                clip_tgz = hf_hub_download(
                    repo_id=dataset_name,
                    revision=config.train.train_policy.dataset_revision or None,
                    repo_type="dataset",
                    filename=clip_filename,
                )
            _extract_tgz_file(clip_tgz, video_clips_dir)

    if config.train.train_policy.dataset_subset == "av":
        # For AV dataset, we need to rename the files
        for root, dirs, files in os.walk(video_clips_dir):
            for file in files:
                if file.endswith(".mp4"):
                    new_name = file.split(".")[0] + ".mp4"
                    os.rename(os.path.join(root, file), os.path.join(root, new_name))
                    logger.info(f"Renamed {file} to {new_name}")


GPU_FLOPS_MAPPING = {
    "NVIDIA H100 80GB HBM3": {
        "FP32": 989 * (1e12),
        "FP16": 1979 * (1e12),
    },
    "NVIDIA A100 80GB": {
        "FP32": 39 * (1e12),
        "FP16": 195 * (1e12),
    },
}


def get_device_flops(dtype: torch.dtype, num_gpus: int) -> int:
    """
    Get the GPU FLOPs for the current device.

    Args:
        dtype (torch.dtype): The data type of the model.
        num_gpus (int): The number of GPUs available.

    Returns:
        int: The FLOPs of the current device.
    """
    gpu_flops = 0
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_type = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        if dtype == "float32":
            gpu_flops = GPU_FLOPS_MAPPING[gpu_type]["FP32"]
        elif dtype == "float16" or dtype == "bfloat16":
            gpu_flops = GPU_FLOPS_MAPPING[gpu_type]["FP16"]
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    else:
        logger.warning("CUDA is not available. Cannot get GPU FLOPs.")
    return gpu_flops * num_gpus


def compute_mfu(
    model: torch.nn.Module,
    n_tokens: int,
    iter_time: float,
    num_gpus: int,
    dtype: str,
) -> dict:
    """
    Compute the model FLOPs Utilization (MFU) for a given model and number of tokens.

    Args:
        model (torch.nn.Module): The model for which to compute the MFU.
        n_tokens (int): The number of tokens in the inputs.
        iter_time (float): The time taken for the forward pass in seconds.

    Returns:
        dict: A dictionary containing the MFU/FLOPs value.
    """
    result = {}
    # Get the model FLOPs for the iteration
    model_params, model_flops = model.get_nparams_and_flops(n_tokens)
    result["model_flops"] = model_flops

    # Get the GPU FLOPs
    try:
        gpu_flops = get_device_flops(dtype, num_gpus)
        # Calculate and return the MFU
        mfu = (model_flops / gpu_flops) / iter_time
        result["mfu"] = mfu
        logger.info(
            f"MFU: {mfu:.4f}, Model FLOPs: {model_flops / 1e12:.2f}T, GPU FLOPs: {gpu_flops / 1e12:.2f}T, Iter time: {iter_time:.2f}s"
        )
    except Exception as e:
        logger.error(
            f"Cannot compute MFU: {e}, only report model FLOPs, you can calculate MFU manually."
        )
        logger.info(
            f"Model FLOPs: {model_flops / 1e12:.2f}T, Iter time: {iter_time:.2f}s"
        )

    return result


def fix_data_type_size(obj):
    if isinstance(obj, tuple):
        return tuple([fix_data_type_size(x) for x in obj])
    elif isinstance(obj, list):
        return [fix_data_type_size(x) for x in obj]
    elif isinstance(obj, dict):
        return {fix_data_type_size(k): fix_data_type_size(v) for k, v in obj.items()}
    elif isinstance(obj, int):
        return ctypes.c_int64(obj)
    else:
        return obj


# Extension code
MSGPACK_C_LONG_EXT_TYPE = 0x10


# Hooks
def msgpack_c_long(obj):
    if isinstance(obj, ctypes._SimpleCData) and isinstance(obj.value, int):
        if isinstance(obj, ctypes.c_long):
            return ExtType(
                MSGPACK_C_LONG_EXT_TYPE, obj.value.to_bytes(8, "big", signed=True)
            )
    raise TypeError(f"Unsupported type: {type(obj)}")


def msgunpack_c_long(code, data):
    if code == MSGPACK_C_LONG_EXT_TYPE:
        val = int.from_bytes(data, "big", signed=True)
        return int(val)
    return ExtType(code, data)


def sync_model_vocab(
    model_name_or_path,
    lm_head_key="lm_head.weight",
    embed_tokens_key="model.embed_tokens.weight",
):
    self_rank = int(os.environ.get("RANK", 0))
    vocab_size = None
    if self_rank == 0:
        weight_map_path = resolve_model_path(
            f"{model_name_or_path}:model.safetensors.index.json"
        )
        weight_map = read_json_file(weight_map_path)["weight_map"]

        if lm_head_key in weight_map:
            lm_head = weight_map[lm_head_key]
            lm_head_path = resolve_model_path(f"{model_name_or_path}:{lm_head}")
            with safe_open(lm_head_path, framework="pt", device="cpu") as f:
                tensor_slice = f.get_slice(lm_head_key)
                vocab_size, _ = tensor_slice.get_shape()
        elif embed_tokens_key in weight_map:
            embed_tokens = weight_map[embed_tokens_key]
            embed_tokens_path = resolve_model_path(
                f"{model_name_or_path}:{embed_tokens}"
            )
            with safe_open(embed_tokens_path, framework="pt", device="cpu") as f:
                tensor_slice = f.get_slice(embed_tokens_key)
                vocab_size, _ = tensor_slice.get_shape()
        else:
            raise ValueError(
                "Could not find `lm_head` or `model.embed_tokens.weight` in the model."
            )
    from cosmos_reason1.utils.distributed import broadcast_object_cpu

    vocab_size = broadcast_object_cpu(vocab_size, src=0, device=torch.device("cpu"))
    logger.info(f"Vocabulary size: {vocab_size}")

    return vocab_size


def retry(func=None, *, max_retry=10, max_delay=30.0):
    """
    Decorator (or wrapper) to retry a function up to max_retry times,
    backing off exponentially (1s, 2s, 4s, …) up to max_delay seconds.

    Usage:

      @retry(max_retry=5)                # uses default max_delay=30
      def foo(...): …

      @retry(max_retry=5, max_delay=60)  # override max_delay
      def bar(...): …

      wrapped = retry(baz, max_retry=2)  # direct call style
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            delay = 1.0
            for attempt in range(max_retry + 1):
                try:
                    return f(*args, **kwargs)
                except Exception:
                    if attempt == max_retry:
                        # out of retries: re-raise last exception
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)

        return wrapper

    # allow both @retry(...) and retry(func, ...)
    if callable(func):
        return decorator(func)
    return decorator


def seperate_nccl_comm_needed():
    """
    Check if separate NCCL communications needed to prevent hang.
    """
    return False


def write_redis_config(port, logfile, file_path="/opt/redis_config.conf"):
    """
    Write the redis config file.
    redis_config_path: the path to the redis config file.
    port: the port for Redis to listen on.
    logfile: the logfile for Redis.

    return the actual path of the redis config file.
    """
    config_content = f"""# Redis configuration file example for insecure connections

# Bind to all network interfaces (use with caution)
bind 0.0.0.0

# Set the port for Redis to listen on (default is {port})
port {port}

# Disable TLS by setting the tls-port to 0
tls-port 0

# Disable authentication by commenting out the requirepass directive
# requirepass yourpassword

# Other configuration settings can remain as default or be customized as needed
timeout 0
tcp-keepalive 300
protected-mode no
# enable-protected-configs yes
# enable-debug-command yes
# enable-module-command yes
daemonize yes
supervised no
loglevel notice
logfile {logfile}
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /opt
"""
    with open(file_path, "w") as file:
        file.write(config_content)
    return file_path


def do_once(func):
    """Decorator to make a function only call once."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = func.__name__ + str(args) + str(kwargs)
        if key not in do_once.dones:
            do_once.dones.add(key)
            return func(*args, **kwargs)

    return wrapper


do_once.dones = set()


def is_master_rank(parallel_dims, global_rank: int):
    return (not parallel_dims.pp_enabled and global_rank == 0) or (
        parallel_dims.pp_enabled
        and global_rank
        == (parallel_dims.world_size - parallel_dims.world_size / parallel_dims.pp)
    )
