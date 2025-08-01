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
import tarfile
import argparse
from huggingface_hub import hf_hub_download, login, HfApi
from typing import Optional, List
from collections import defaultdict
import logging as log


def download_repository(repo_id: str, target_dir: str, repo_type: str = "dataset",
                      hf_token: Optional[str] = None, revision: str = "main",
                      file_paths: List[str] = None):
    """
    Download files from a Hugging Face repository and save them to the target directory.

    Args:
        repo_id: The Hugging Face repository ID
        target_dir: The target directory where files will be saved
        repo_type: Repository type ('dataset' or 'model')
        hf_token: Hugging Face API token for private repositories
        revision: The repository revision to download from (branch name, tag, or commit hash)
        file_paths: Specific file paths to download (if None, will discover structure)
    """
    # Authenticate with Hugging Face
    if hf_token:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face")
    else:
        print("No Hugging Face token provided. Attempting to use cached credentials...")

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # For models, create a subfolder with the model name (e.g., Cosmos-Reason1-7B)
    if repo_type == "model":
        target_dir = os.path.join(target_dir, repo_id)
        os.makedirs(target_dir, exist_ok=True)
        print(f"Model files will be saved in: {target_dir}")

    # Initialize API
    api = HfApi(token=hf_token)

    try:
        print(f"Accessing repository: {repo_id}")

        # List repository contents
        files = api.list_repo_files(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type
        )

        print(f"Successfully accessed repository {repo_id}! Found {len(files)} files.")

        # If no specific file paths are provided, discover files based on repo_type
        if file_paths is None:
            if repo_type == "model":
                # Select model-related files
                file_paths = [
                    f for f in files
                    if f.endswith(('.safetensors', '.json', '.bin', '.pt', '.model'))
                    or f in ('.gitattributes', 'README.md')
                ]
            else:  # repo_type == "dataset"
                # Select dataset-related files (JSON and archives)
                file_paths = [
                    f for f in files
                    if f.endswith(('_benchmark_qa_pairs.json', '.tar.gz', '.json'))
                    or f in ('.gitattributes', 'README.md')
                ]

            print("\nWill process these files based on repository structure:")
            for path in file_paths:
                print(f"- {path}")

        if not file_paths:
            print("\nNo suitable files were found in this repository.")
            return

        # Download files
        for file_path in file_paths:
            try:
                # Extract directory part from file path
                dir_parts = os.path.dirname(file_path).split('/')
                if dir_parts and dir_parts[0]:
                    subdir_path = os.path.join(target_dir, *dir_parts)
                    os.makedirs(subdir_path, exist_ok=True)
                else:
                    subdir_path = target_dir

                # Construct local file path
                file_name = os.path.basename(file_path)
                local_file_path = os.path.join(subdir_path, file_name)

                # Simple check if file already exists
                if os.path.exists(local_file_path):
                    print(f"\nSkipping {file_path} as it already exists at {local_file_path}")
                    continue

                print(f"\nDownloading {file_path}...")

                # Download file from Hugging Face
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    token=hf_token,
                    revision=revision,
                    repo_type=repo_type,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False
                )

                # Check if the file is an archive that needs extraction (for datasets)
                if repo_type == "dataset" and file_path.endswith('.tar.gz') and os.path.exists(downloaded_file):
                    print(f"Extracting {file_name} to {subdir_path}...")
                    with tarfile.open(downloaded_file, 'r:gz') as tar:
                        tar.extractall(path=subdir_path)
                    print(f"Extraction of {file_name} complete.")

                print(f"Successfully downloaded {file_name} to {subdir_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    except Exception as e:
        print(f"Error accessing repository: {str(e)}")
        print("\nPlease verify the repository exists and your token has access.")
        return


def main():
    parser = argparse.ArgumentParser(description="Download files from Hugging Face repository")
    parser.add_argument("--task", type=str, choices=["benchmark", "sft", "rl"])
    parser.add_argument("--repo", type=str, help="Hugging Face repository ID")
    parser.add_argument("--target", type=str, required=True, help="Target directory for extraction")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    parser.add_argument("--type", type=str, default="dataset", choices=["dataset", "model"], help="Repository type")
    parser.add_argument("--revision", type=str, default="main", help="Repository revision (branch, tag, commit)")
    parser.add_argument("--file", action="append", help="Specific file paths to download (can be used multiple times)")
    args = parser.parse_args()

    hf_map = {
        "benchmark": "nvidia/Cosmos-Reason1-Benchmark",
        "sft": "nvidia/Cosmos-Reason1-SFT-Dataset",
        "rl": "nvidia/Cosmos-Reason1-RL-Dataset",
    }

    assert args.task or args.repo, "Either --task or --repo must be provided."
    if args.task:
        assert args.task in hf_map, f"Task {args.task} not found in the mapping."
        args.repo = hf_map[args.task]

    # If token is not provided via arguments, try to get it from environment variable or ask user
    hf_token = args.token
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            log.warning("No Hugging Face token (HF_TOKEN) provided via args or environment.")

    # Download files from repository
    download_repository(
        repo_id=args.repo,
        target_dir=f"{args.target}/{args.task}",
        repo_type=args.type,
        hf_token=hf_token,
        revision=args.revision,
        file_paths=args.file
    )

    print(f"Repository download process completed. Files saved to {args.target}")


if __name__ == "__main__":
    main()
