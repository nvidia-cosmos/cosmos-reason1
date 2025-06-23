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
from typing import Any
from cosmos_reason1.utils.logging import logger
import os
import shutil


def modelscope_download_model(name: str, **kwargs):
    """
    Download a model from ModelScope.
    Args:
        name (str): The name of the model to download.
        **kwargs: Additional arguments to pass to the snapshot_download function.
    """
    cache_dir = os.environ.get(
        "MODELSCOPE_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache/modelscope_cache/model"),
    )
    local_dir = os.path.join(cache_dir, name)
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            if not os.path.exists(local_dir):
                snapshot_download(
                    name,
                    local_dir=local_dir,
                    repo_type="model",
                    max_workers=16,
                    **kwargs,
                )
    except Exception as e:
        logger.error(f"Failed to download model {name} from ModelScope: {e}")
        # Remove all in foler local_dir
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)

    return local_dir


def modelscope_download_dataset(dataset_name: str, **kwargs):
    """
    Download a dataset from ModelScope.
    Args:
        name (str): The name of the dataset to download.
        **kwargs: Additional arguments to pass to the snapshot_download function.
    """
    cache_dir = os.environ.get(
        "MODELSCOPE_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache/modelscope_cache/dataset"),
    )
    local_dir = os.path.join(cache_dir, dataset_name)
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            if not os.path.exists(local_dir):
                allow_patterns = None
                if "subset_name" in kwargs:
                    subset_name = kwargs.pop("subset_name")
                    allow_patterns = [f"{subset_name}/*", f"{subset_name}/**/*", "*"]
                snapshot_download(
                    dataset_name,
                    local_dir=local_dir,
                    repo_type="dataset",
                    max_workers=16,
                    allow_patterns=allow_patterns,
                    **kwargs,
                )
    except Exception as e:
        logger.error(f"Failed to download dataset {dataset_name} from ModelScope: {e}")
        # Remove all in foler local_dir
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        if os.path.exists(os.path.join(local_dir, "dataset_infos.json")):
            os.remove(os.path.join(local_dir, "dataset_infos.json"))

    return local_dir


def modelscope_download(name: str, repo_type: str = "model", **kwargs):
    """
    Download a model or dataset from ModelScope.

    Args:
        name (str): The name of the model or dataset to download.
        repo_type (str): The type of repository, either 'model' or 'dataset'.
        **kwargs: Additional arguments to pass to the snapshot_download function.
    """
    if os.environ.get("COSMOS_USE_MODELSCOPE", False) in [True, "True", "true", "1"]:
        if repo_type == "model":
            return modelscope_download_model(name, **kwargs)
        elif repo_type == "dataset":
            return modelscope_download_dataset(name, **kwargs)
    return name


def update_config_if_modelscope(loaded_config: Any):
    """
    Update the model and dataset paths in the loaded configuration to point to the modelscope storage.
    This is necessary for compatibility with the modelscope storage system.
    """
    loaded_config.policy.model_name_or_path = modelscope_download(
        loaded_config.policy.model_name_or_path, "model"
    )
    loaded_config.train.train_policy.dataset.name = modelscope_download(
        loaded_config.train.train_policy.dataset.name,
        "dataset",
        subset_name=loaded_config.train.train_policy.dataset.subset,
        revision=loaded_config.train.train_policy.dataset.revision,
    )

    if loaded_config.validation.dataset.name:
        loaded_config.validation.dataset.name = modelscope_download(
            loaded_config.validation.dataset.name,
            "dataset",
            subset_name=loaded_config.validation.dataset.subset,
            revision=loaded_config.validation.dataset.revision,
        )

    return loaded_config


def modelscope_load_dataset(dataset_name: str, subset_name: str, split: str):
    """
    Load dataset from ModelScope.
    """
    if os.environ.get("COSMOS_USE_MODELSCOPE", False) in [True, "True", "true", "1"]:
        from modelscope.msdatasets import MsDataset

        if os.path.exists(dataset_name):
            from datasets import load_dataset

            return load_dataset(dataset_name, subset_name, split=split)
        return MsDataset.load(dataset_name, subset_name=subset_name, split=split)
    return None
