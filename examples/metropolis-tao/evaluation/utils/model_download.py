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

import logging as log
import os

from huggingface_hub import hf_hub_download, list_repo_files

# Define a dictionary mapping model names to Hugging Face repository IDs.
# This helps manage repository mappings centrally.
MODEL_REPO_MAP: dict[str, str] = {
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
}

# Define a list of standard tokenizer filenames to download.
TOKENIZER_FILENAMES: list[str] = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "chat_template.json",
]


def check_model_shards_complete(model_dir: str) -> bool:
    """
    Check if all model shards are completely downloaded in the specified directory.

    For files following the pattern "model-NNNNN-of-MMMMM.safetensors",
    this function verifies that all expected shards (from 00001 to MMMMM) are present.

    Args:
        model_dir: The directory path where model files are stored.

    Returns:
        bool: True if all expected shards are found based on the naming pattern,
              False otherwise or if no shards are found.
    """
    # List all items in the target directory.
    try:
        files = os.listdir(model_dir)
    except FileNotFoundError:
        log.error(f"Model directory not found: {model_dir}")
        return False
    except Exception as e:
        log.error(f"Error listing directory {model_dir}: {e}")
        return False

    # Filter for files that appear to be model shards based on naming convention.
    shard_files = [
        f for f in files if f.startswith("model-") and f.endswith(".safetensors")
    ]

    # If no files matching the shard pattern are found, the check fails.
    if not shard_files:
        log.warning(f"No model shard files found in {model_dir}")
        return False

    total_shards = -1
    # Attempt to extract the total number of shards from any shard filename.
    # The format is expected to be "model-#####-of-#####.safetensors".
    for f in shard_files:
        parts = f.split("-of-")
        if len(parts) == 2:
            try:
                # Extract the total count from the second part before the extension.
                total_shards = int(parts[1].split(".")[0])
                break  # Found the total count, no need to check other files.
            except ValueError:
                # Ignore files that match the pattern but have non-integer shard counts.
                continue

    # If total_shards could not be determined from any file, the check fails.
    if total_shards <= 0:
        log.error(
            f"Could not determine the total number of shards from files in {model_dir}"
        )
        return False

    # Generate the set of all expected shard filenames based on the determined total.
    expected_shards = set(
        f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
        for i in range(total_shards)
    )
    # Create a set of the actual shard filenames found.
    actual_shards = set(shard_files)

    # Check if all expected shards are present in the directory.
    is_complete = expected_shards.issubset(actual_shards)

    if is_complete:
        log.info(f"All {total_shards} model shards found in {model_dir}")
    else:
        missing_shards = expected_shards - actual_shards
        log.warning(
            f"Missing {len(missing_shards)}/{total_shards} model shards in {model_dir}."
        )
        # Optionally log missing files, be cautious with large numbers.
        if len(missing_shards) < 10:
            log.warning(f"Missing files: {missing_shards}")

    return is_complete


def download_checkpoint(repo_id: str, checkpoint_output_dir: str) -> None:
    """
    Download all checkpoint files from a Hugging Face model repository to a specified output directory.

    This function lists all files available in the given model's repository on Hugging Face
    and downloads them locally, skipping files that already exist.

    Args:
        repo_id: The Hugging Face repository ID (e.g., "nvidia/Cosmos-Reason1-7B").
        checkpoint_output_dir: The local directory where checkpoint files will be saved.

    Raises:
        ValueError: If the provided model name is not recognized.
    """
    if check_model_shards_complete(checkpoint_output_dir):
        return

    log.info(
        f"Attempting to download all checkpoint files for {repo_id} to {checkpoint_output_dir}"
    )

    os.makedirs(checkpoint_output_dir, exist_ok=True)

    try:
        all_files = list_repo_files(repo_id)
    except Exception as e:
        log.error(f"Failed to list files for repository {repo_id}: {e}")
        return

    for filename in all_files:
        file_path = os.path.join(checkpoint_output_dir, filename)
        if os.path.exists(file_path):
            log.debug(f"File already exists: {filename}")
            continue

        log.info(f"Downloading file: {filename}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=checkpoint_output_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            log.error(f"Failed to download file {filename} from {repo_id}: {e}")

    log.info(
        f"Finished downloading all checkpoint files for {repo_id} to {checkpoint_output_dir}"
    )


def download_tokenizer(model: str, checkpoint_output_dir: str) -> None:
    """
    Download standard tokenizer files for a given model to a specified output directory.

    This function uses the predefined mapping from model name to Hugging Face
    repository ID and downloads a standard set of tokenizer configuration files.
    It skips downloading files that already exist locally.

    Args:
        model: The name of the model (used to look up the repository ID).
        checkpoint_output_dir: The local directory where tokenizer files will be saved.

    Raises:
        ValueError: If the provided model name is not recognized.
    """
    log.info(
        f"Attempting to download tokenizer files for {model} to {checkpoint_output_dir}"
    )

    # Look up the Hugging Face repository ID based on the model name.
    repo_id = MODEL_REPO_MAP.get(model)

    # If the model name is not in the mapping, raise an error.
    if repo_id is None:
        raise ValueError(
            f"Invalid or unsupported model '{model}'. "
            f"Supported models are: {list(MODEL_REPO_MAP.keys())}"
        )

    # Ensure the output directory exists before downloading.
    os.makedirs(checkpoint_output_dir, exist_ok=True)

    # Iterate through the list of required tokenizer filenames.
    for filename in TOKENIZER_FILENAMES:
        file_path = os.path.join(checkpoint_output_dir, filename)
        # Check if the file already exists to avoid unnecessary re-downloads.
        if os.path.exists(file_path):
            log.debug(f"Tokenizer file already exists: {filename}")
            continue  # Skip download if file is present.

        log.info(f"Downloading tokenizer file: {filename}")
        try:
            # Download the file from the Hugging Face Hub.
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=checkpoint_output_dir,
                local_dir_use_symlinks=False,  # Use False for direct file download
            )
        except Exception as e:
            log.error(
                f"Failed to download tokenizer file {filename} from {repo_id}: {e}"
            )
            # Depending on requirements, you might want to raise the exception
            # or continue attempting to download other files. Continuing here.

    log.info(
        f"Finished processing tokenizer files for {model} in {checkpoint_output_dir}"
    )
