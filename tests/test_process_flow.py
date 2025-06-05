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
import pytest
import subprocess
import sys
from cosmos_reason1.utils import util
import toml
import tempfile


def test_process_exit_grpo():
    """Test grpo all processes exit cleanly."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    world_size = 2
    tools_dir = os.path.join(cur_dir, "..", "tools")
    port = util.find_available_port(8123)
    config_path = os.path.join(
        cur_dir,
        "..",
        "configs",
        "qwen2-5",
        "qwen2-5-3b-p-fsdp1-tp2-r-tp2-pp1-grpo.toml",
    )
    with open(config_path, "r") as f:
        config = toml.load(f)
    config["train"]["epoch"] = 1
    config["train"]["train_policy"]["dataset_name"] = os.path.join(
        cur_dir, "test_dataset"
    )
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".toml", delete=False
    ) as tmpfile:
        toml.dump(config, tmpfile)
        tmpfile_toml = tmpfile.name
    controller_cmd = (
        f"{os.path.join(tools_dir, "launch_controller.sh")} --config {tmpfile_toml}"
    )
    controller_cmd += f" --port {port}"
    controller_process = subprocess.Popen(
        controller_cmd,
        shell=True,
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{port}"
    os.environ["COSMOS_CONTROLLER_LAUNCHER"] = "cosmos_reason1.dispatcher.run_web_panel"
    # Create the Python command for torchrun
    policy_cmd = [
        "torchrun",
        f"--nproc_per_node={world_size}",  # Use 2 GPUs
        "--role=rank",
        "--tee=3",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        os.path.join(cur_dir, "launch_test_worker.py"),
        "-1",
        "-1",
        "dummy_policy",
    ]
    rollout_cmd = [
        "torchrun",
        f"--nproc_per_node={world_size}",  # Use 2 GPUs
        "--role=rank",
        "--tee=3",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        os.path.join(cur_dir, "launch_test_worker.py"),
        "-1",
        "-1",
        "dummy_rollout",
    ]
    policy_env = dict(os.environ)
    policy_env["CUDA_VISIBLE_DEVICES"] = "0,1"
    # Start the process
    policy_process = subprocess.Popen(
        policy_cmd,
        stdout=sys.stderr,
        stderr=sys.stderr,
        env=policy_env,
    )
    rollout_env = dict(os.environ)
    rollout_env["CUDA_VISIBLE_DEVICES"] = "2,3"
    rollout_process = subprocess.Popen(
        rollout_cmd,
        stdout=sys.stderr,
        stderr=sys.stderr,
        env=rollout_env,
    )

    processes = [controller_process, policy_process, rollout_process]

    # Wait for process to complete
    for process in processes:
        stdout, stderr = process.communicate()
        # Check if process completed successfully
        assert (
            process.returncode == 0
        ), f"Process failed with code: {process.returncode}"


def test_process_exit_sft():
    """Test sft all processes exit cleanly."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    world_size = 2
    tools_dir = os.path.join(cur_dir, "..", "tools")
    port = util.find_available_port(8123)
    config_path = os.path.join(
        cur_dir,
        "..",
        "configs",
        "qwen2-5",
        "qwen2-5-3b-tp2-fsdp-sft.toml",
    )
    with open(config_path, "r") as f:
        config = toml.load(f)
    config["train"]["epoch"] = 1
    config["train"]["train_policy"]["dataset_name"] = os.path.join(
        cur_dir, "test_dataset"
    )
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".toml", delete=False
    ) as tmpfile:
        toml.dump(config, tmpfile)
        tmpfile_toml = tmpfile.name
    controller_cmd = (
        f"{os.path.join(tools_dir, "launch_controller.sh")} --config {tmpfile_toml}"
    )
    controller_cmd += f" --port {port}"
    controller_process = subprocess.Popen(
        controller_cmd,
        shell=True,
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{port}"
    os.environ["COSMOS_CONTROLLER_LAUNCHER"] = "cosmos_reason1.dispatcher.run_web_panel"
    # Create the Python command for torchrun
    policy_cmd = [
        "torchrun",
        f"--nproc_per_node={world_size}",  # Use 2 GPUs
        "--role=rank",
        "--tee=3",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        os.path.join(cur_dir, "launch_test_worker.py"),
        "-1",
        "-1",
        "dummy_policy",
    ]
    policy_env = dict(os.environ)
    policy_env["CUDA_VISIBLE_DEVICES"] = "0,1"
    # Start the process
    policy_process = subprocess.Popen(
        policy_cmd,
        stdout=sys.stderr,
        stderr=sys.stderr,
        env=policy_env,
    )
    processes = [controller_process, policy_process]

    # Wait for process to complete
    for process in processes:
        stdout, stderr = process.communicate()
        # Check if process completed successfully
        assert (
            process.returncode == 0
        ), f"Process failed with code: {process.returncode}"


if __name__ == "__main__":
    pytest.main([__file__])
