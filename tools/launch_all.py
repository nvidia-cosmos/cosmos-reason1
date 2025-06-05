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

#!/usr/bin/env python3

import socket
import subprocess
import sys
import logging
import time
import os
import re
import argparse
from typing import List, Dict, Optional, Any, Callable
import toml
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cosmos")

COSMOS_REASON1_DIR = "/workspace/cosmos_reason1"
TOOLS_RELATIVE_DIR = "tools"


def wait_for_url_ready(url: str):
    """
    Wait for a URL to be ready by sending a GET request.

    Args:
        url: The URL to check

    Returns:
        None
    """
    while True:
        # create TCP socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            host, port = url.split(":")
            sock.connect((host, int(port)))
            sock.close()
            break
        except socket.error:
            # If the connection fails, wait and retry
            time.sleep(1)


def read_config(config_file: str) -> Dict[str, Any]:
    """
    Read configuration from a TOML file.

    Args:
        config_file: Path to the TOML configuration file

    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_file, "r") as f:
            config = toml.load(f)
        return config
    except Exception as e:
        logger.error(f"Error reading config file {config_file}: {e}")
        sys.exit(1)


def get_available_gpus() -> List[str]:
    """
    Detect available GPUs using nvidia-smi and return their IDs.

    Returns:
        List of GPU IDs as strings
    """
    try:
        cmd = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        cvd = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if cvd is not None:
            # Add the GPU IDs to the command
            cmd += ["--id=" + cvd]
        # Run nvidia-smi to get GPU information
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse the output to get GPU IDs
        gpu_ids = [line.strip() for line in result.stdout.splitlines()]

        if not gpu_ids:
            logger.error("Warning: No GPUs detected")
            return []

        logger.info(f"Detected {len(gpu_ids)} GPUs: {', '.join(gpu_ids)}")
        return gpu_ids

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running nvidia-smi: {e}")
        return []
    except Exception as e:
        logger.error(f"Error detecting GPUs: {e}")
        return []


def launch_processes(
    commands: List[str],
    gpu_devices: Optional[List[str]],
    control_urls: Optional[List[str]],
    output_files: Optional[List[str]],
) -> List[subprocess.Popen]:
    """
    Launch multiple subprocesses and return their process objects.

    Args:
        commands: List of command strings to execute
        gpu_devices: List of GPU device IDs to assign to each process (e.g., ["0", "1", "2"])
        control_urls: List of controller URLs to assign to each process (e.g., ["localhost:8000"])
        output_files: List of output files to redirect process output to (e.g., ["output1.log", "output2.log"])

    Returns:
        List of Popen objects for the launched processes
    """
    processes = []

    if gpu_devices is None:
        gpu_devices = [None] * len(commands)
    elif len(gpu_devices) != len(commands):
        raise ValueError("Number of GPU devices must match number of commands")

    for cmd, gpu_id, url, ofile in zip(
        commands, gpu_devices, control_urls, output_files
    ):
        try:
            # Prepare environment variables
            env = dict(os.environ)
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
            if url is not None:
                env["COSMOS_CONTROLLER_HOST"] = url

            if ofile is not None:
                f = open(ofile, "wb")
                cout = f
                cerr = f
            else:
                cout = sys.stdout
                cerr = sys.stderr

            # Launch process and capture output
            logger.info(f"Launching process with command: {cmd}")
            process = subprocess.Popen(
                cmd, shell=True, stdout=cout, stderr=cerr, env=env
            )
            processes.append(process)
            if ofile is not None:
                f.close()
        except Exception as e:
            logger.error(f"Error launching process for command '{cmd}': {e}")

    return processes


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch multiple processes with GPU assignments"
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default="cosmos_reason1.dispatcher.run_web_panel",
        help="The launcher to use, default is `cosmos_reason1.dispatcher.run_web_panel`, a custom launcher can be provided for custom dataset and reward functions injection.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TOML configuration file, which specifies the detailed configuration for the whole training process including algorithm, model, data, parallelism, etc.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL of the controller for the policy and rollout replicas to connect to, consisting of IP and port in the format ip:port. If not provided, the controller will be launched on the local machine. If provided and the IP is the local IP, the controller will be launched on the local machine. If provided and the IP is not the local IP, the controller will be launched on the remote machine.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port of the controller to connect to, default is 8000. This is only used when --url is not provided to launch the controller on the local machine.",
    )
    parser.add_argument(
        "--policy",
        type=int,
        default=None,
        help="Total number of policy replicas to launch in the whole system. If not provided, the number of policy replicas will be obtained from TOML configuration file.",
    )
    parser.add_argument(
        "--rollout",
        type=int,
        default=None,
        help="Total number of rollout replicas to launch in the whole system. If not provided, the number of rollout replicas will be obtained from TOML configuration file.",
    )
    parser.add_argument(
        "--p2r-ratio",
        type=str,
        default=None,
        help="Ratio of policy replicas to rollout replicas. This is used to determine the number of rollout replicas and the number of policy replicas based on the number of workers.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save logs. If not provided, logs will be printed to stdout.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers to use for the job, default is 1. This is used when multi-node training are used for the job.",
    )
    parser.add_argument(
        "--worker-idx",
        type=int,
        default=0,
        help="Worker index for local execution. In Lepton mode, this is ignored as worker indices are automatically assigned by Lepton.",
    )

    parser.add_argument(
        "--node-ip-list",
        type=str,
        default=None,
        help="list of ips for all the workers, separated by ';'. This is used when multi-node training are used for one replica.",
    )

    parser.add_argument(
        "--rdzv-port",
        type=int,
        default=29345,
        help="Rendezvous endpoint port for the job, default is 29345. This is used when multi-node training are used for one replica.", 
    )

    parser.add_argument(
        "--lepton-mode",
        action="store_true",
        default=False,
        help="Enable Lepton mode for remote execution",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable log level to debug",
    )

    # Lepton specific options
    lepton_group = parser.add_argument_group("Lepton mode options")
    lepton_group.add_argument("--lepton-job-name", "-n", type=str, help="Job name")
    lepton_group.add_argument(
        "--lepton-container-image", type=str, help="Container image for the job"
    )
    lepton_group.add_argument(
        "--lepton-container-port",
        type=str,
        help="Ports to expose for the job, in the format portnumber[:protocol]",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-resource-shape", type=str, help="Resource shape for the pod"
    )
    lepton_group.add_argument(
        "--lepton-node-group",
        "-ng",
        type=str,
        help="Node group for the job",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-max-failure-retry",
        type=int,
        help="Maximum number of failures to retry per worker",
    )
    lepton_group.add_argument(
        "--lepton-max-job-failure-retry",
        type=int,
        help="Maximum number of failures to retry per whole job",
    )
    lepton_group.add_argument(
        "--lepton-env",
        "-e",
        type=str,
        help="Environment variables to pass to the job, in the format `NAME=VALUE`",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-secret",
        "-s",
        type=str,
        help="Secrets to pass to the job",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-mount",
        type=str,
        help="Persistent storage to be mounted to the job",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-image-pull-secrets",
        type=str,
        help="Secrets to use for pulling images",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-intra-job-communication",
        type=bool,
        help="Enable intra-job communication",
    )
    lepton_group.add_argument(
        "--lepton-privileged",
        action="store_true",
        help="Run the job in privileged mode",
    )
    lepton_group.add_argument(
        "--lepton-ttl-seconds-after-finished",
        type=int,
        help="TTL for finished jobs in seconds",
        default=259200,
    )
    lepton_group.add_argument(
        "--lepton-log-collection",
        "-lg",
        type=bool,
        help="Enable or disable log collection",
    )
    lepton_group.add_argument(
        "--lepton-node-id", "-ni", type=str, help="Node for the job", action="append"
    )
    lepton_group.add_argument(
        "--lepton-queue-priority", "-qp", type=str, help="Queue priority for the job"
    )
    lepton_group.add_argument(
        "--lepton-visibility", type=str, help="Visibility of the job (public/private)"
    )
    lepton_group.add_argument(
        "--lepton-shared-memory-size", type=int, help="Shared memory size in MiB"
    )
    lepton_group.add_argument(
        "--lepton-with-reservation",
        type=str,
        help="Reservation ID for dedicated node groups",
    )

    args = parser.parse_args()

    # Validate Lepton mode arguments
    if args.lepton_mode:
        required_args = [("lepton_job_name", "--lepton-job-name")]

        for arg_name, arg_flag in required_args:
            if not getattr(args, arg_name):
                parser.error(f"{arg_flag} is required when --lepton-mode is enabled")

    return args, parser


def get_local_ip():
    """
    Get the local IP address of the machine.

    Returns:
        Local IP address as a string
    """
    try:
        import socket

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return [local_ip, hostname]
    except Exception as e:
        logger.error(f"Error getting local IP address: {e}")
        return None

def replica_placement(
    available_gpus: List[int],
    n_policy: int,
    n_rollouts: int,
    min_n_gpus_policy: int,
    min_n_gpus_rollout: int,
    replica_script: str,
    control_url: str,
    output_dir: Optional[str],
    get_worker_ip: Optional[Callable] = None,
    rdzv_port: Optional[int] = None,
) -> List[List[str]]:
    commands = []
    gpu_devices = []
    control_urls = []
    output_files = []
    assert len(available_gpus) in [1, 2, 4, 8], "Number of GPUs per worker must be 1, 2, 4, or 8"
    # Prepare the command to launch the controller for all workers
    global_available_gpus = [available_gpus]
    # Create commands for policy and rollout replicas
    gpu_idx = 0
    global_worker_idx = 0
    global_launch_settings = []

    # Assign launch settings for each worker
    for i in range(n_policy):
        if min_n_gpus_policy > len(global_available_gpus[global_worker_idx]):
            assert min_n_gpus_policy % len(
                global_available_gpus[global_worker_idx]
            ) == 0, f"min_n_gpus_policy {min_n_gpus_policy} is not divisible by {len(global_available_gpus[global_worker_idx])}"
            nodes_needed = min_n_gpus_policy // len(
                global_available_gpus[global_worker_idx]
            )
            rdzv_ip = "localhost"
            for node_in_replica in range(nodes_needed):
                gpu_devices.append(
                    ",".join(
                        [
                            str(g)
                            for g in global_available_gpus[global_worker_idx]
                        ]
                    )
                )
                commands.append(f"{replica_script} --type policy --ngpus {len(global_available_gpus[global_worker_idx])} --nnodes {nodes_needed}")
                if node_in_replica == 0:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"
                    if get_worker_ip is not None:
                        rdzv_ip = get_worker_ip(global_worker_idx)
                else:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"

                control_urls.append(control_url)
                output_files.append(
                    os.path.join(output_dir, f"policy_{i}.log")
                    if output_dir is not None
                    else None
                )
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                global_worker_idx += 1
                global_available_gpus.append(
                    available_gpus)
        else:
            if gpu_idx + min_n_gpus_policy > len(global_available_gpus[global_worker_idx]):
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                gpu_idx = 0
                global_worker_idx += 1
                global_available_gpus.append(
                    available_gpus
                )

            gpu_devices.append(
                ",".join(
                    [
                        str(g)
                        for g in global_available_gpus[global_worker_idx][
                            gpu_idx : gpu_idx + min_n_gpus_policy
                        ]
                    ]
                )
            )
            commands.append(f"{replica_script} --type policy --ngpus {min_n_gpus_policy}")
            control_urls.append(control_url)
            output_files.append(
                os.path.join(output_dir, f"policy_{i}.log")
                if output_dir is not None
                else None
            )
            gpu_idx += min_n_gpus_policy

    if min_n_gpus_rollout > len(global_available_gpus[global_worker_idx]):
        # If the number of GPUs needed for rollout is more than available GPUs, we need to allocate a new worker
        if gpu_idx > 0:
            global_launch_settings.append(
                [commands, gpu_devices, control_urls, output_files]
            )
            commands = []
            gpu_devices = []
            control_urls = []
            output_files = []
            gpu_idx = 0
            global_worker_idx += 1
            global_available_gpus.append(
                available_gpus
            )

    for i in range(n_rollouts):
        if min_n_gpus_rollout > len(global_available_gpus[global_worker_idx]):
            assert min_n_gpus_rollout % len(
                global_available_gpus[global_worker_idx]
            ) == 0, f"min_n_gpus_rollout {min_n_gpus_rollout} is not divisible by {len(global_available_gpus[global_worker_idx])}"
            nodes_needed = min_n_gpus_rollout // len(
                global_available_gpus[global_worker_idx]
            )
            rdzv_ip = "localhost"
            for node_in_replica in range(nodes_needed):
                gpu_devices.append(
                    ",".join(
                        [
                            str(g)
                            for g in global_available_gpus[global_worker_idx]
                        ]
                    )
                )
                commands.append(f"{replica_script} --type rollout --ngpus {len(global_available_gpus[global_worker_idx])} --nnodes {nodes_needed}")
                if node_in_replica == 0:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"
                    if get_worker_ip is not None:
                        rdzv_ip = get_worker_ip(global_worker_idx)
                else:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"
                
                control_urls.append(control_url)
                output_files.append(
                    os.path.join(output_dir, f"rollout_{i}.log")
                    if output_dir is not None
                    else None
                )
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                global_worker_idx += 1
                global_available_gpus.append(
                    available_gpus
                )
        else:
            if gpu_idx + min_n_gpus_rollout > len(global_available_gpus[global_worker_idx]):
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                gpu_idx = 0
                global_worker_idx += 1
                global_available_gpus.append(
                    available_gpus
                )

            gpu_devices.append(
                ",".join(
                    [str(g) for g in available_gpus[gpu_idx : gpu_idx + min_n_gpus_rollout]]
                )
            )
            commands.append(f"{replica_script} --type rollout --ngpus {min_n_gpus_rollout}")
            control_urls.append(control_url)
            output_files.append(
                os.path.join(output_dir, f"rollout_{i}.log")
                if output_dir is not None
                else None
            )
            gpu_idx += min_n_gpus_rollout

    if len(commands) > 0:
        global_launch_settings.append(
            [commands, gpu_devices, control_urls, output_files]
        )
    return global_launch_settings


def main():
    args, parser = parse_args()
    if args.debug:
        os.environ["COSMOS_LOG_LEVEL"] = "DEBUG"

    # Check if the config file is provided
    cosmos_config = read_config(args.config)
    # train.train_policy.dataset_name
    dataset_name = cosmos_config.get("train", {}).get("train_policy", {}).get("dataset_name", None)
    if dataset_name and "nvidia/Cosmos-Reason1-SFT-Dataset" in dataset_name:
        launcher = os.path.join(os.path.dirname(__file__), "dataset", "cosmos_sft.py")
    elif dataset_name and "nvidia/Cosmos-Reason1-RL-Dataset" in dataset_name:
        launcher = os.path.join(os.path.dirname(__file__), "dataset", "cosmos_grpo.py")
    else:
        launcher = args.launcher
    
    os.environ["COSMOS_CONTROLLER_LAUNCHER"] = launcher
    # Get the number of GPUs required for policy and rollout
    # and the number of replicas for each
    policy_parallelism = cosmos_config.get("policy", {}).get("parallelism", {})
    rollout_parallelism = cosmos_config.get("rollout", {}).get("parallelism", {})
    # Calculate the minimum number of GPUs required for policy and rollout
    # based on the parallelism settings in the configuration
    # Treat dp_shard_size as 1 if it is not set
    min_n_gpus_policy = (
        policy_parallelism.get("tp_size", 1)
        * policy_parallelism.get("dp_replicate_size", 1)
        * policy_parallelism.get("pp_size", 1)
        * policy_parallelism.get("cp_size", 1)
    )
    min_n_gpus_rollout = (
        rollout_parallelism.get("tp_size", 1)
        * rollout_parallelism.get("dp_replicate_size", 1)
        * rollout_parallelism.get("pp_size", 1)
        * rollout_parallelism.get("cp_size", 1)
    )
    if policy_parallelism.get("dp_shard_size", 1) >= 1:
        min_n_gpus_policy = (
            min_n_gpus_policy * policy_parallelism.get("dp_shard_size", 1)
        )
    if rollout_parallelism.get("dp_shard_size", 1) >= 1:
        min_n_gpus_rollout = (
            min_n_gpus_rollout * rollout_parallelism.get("dp_shard_size", 1)
        )
    if args.p2r_ratio is not None:
        assert args.num_workers is not None, "When using --p2r-ratio, --num-workers must be specified"
        p2r_ratio = args.p2r_ratio.split(":")
        assert len(p2r_ratio) == 2, "Invalid --p2r-ratio format. Use 'policy:rollout' format."
        p_ratio = int(p2r_ratio[0])
        r_ratio = int(p2r_ratio[1])

        if args.lepton_mode:
            match = re.search(r'(8|4|2)x', args.lepton_resource_shape)
            if match:
                num_gpus_per_node = int(match.group(1))
            else:
                num_gpus_per_node = 1
        else:
            num_gpus_per_node = len(get_available_gpus())

        num_per_ratio = args.num_workers * num_gpus_per_node / (p_ratio * min_n_gpus_policy + r_ratio * min_n_gpus_rollout)
        args.policy = int(num_per_ratio * p_ratio)
        args.rollout = int(num_per_ratio * r_ratio)
        assert args.policy >= 1, "Number of policy replicas must be at least 1"
        assert args.policy * min_n_gpus_policy + args.rollout * min_n_gpus_rollout <= args.num_workers * num_gpus_per_node


    if args.policy is None:
        n_policy = policy_parallelism.get("n_init_replicas", 1)
    else:
        n_policy = args.policy
    if args.rollout is None:
        n_rollouts = rollout_parallelism.get("n_init_replicas", 1)
    else:
        n_rollouts = args.rollout

    # If the training type is SFT, set n_rollouts to 0
    if cosmos_config.get("train", {}).get("train_policy", {}).get("type", "grpo") == "sft":
        n_rollouts = 0

    # Handle Lepton mode
    if args.lepton_mode:
        from leptonai.api.v2.client import APIClient
        from leptonai.config import BASE_IMAGE, VALID_SHAPES
        from leptonai.api.v1.types.job import (
            LeptonJob,
            LeptonJobUserSpec,
            LeptonResourceAffinity,
            ReservationConfig,
        )
        from leptonai.api.v1.types.deployment import (
            LeptonLog,
            QueueConfig,
        )
        from leptonai.api.v1.types.common import Metadata, LeptonVisibility
        from leptonai.api.v1.photon import (
            make_env_vars_from_strings,
            make_mounts_from_strings,
        )
        from leptonai.cli.util import _get_valid_nodegroup_ids, _get_valid_node_ids

        from leptonai.cli.job import make_container_port_from_string

        # Initialize Lepton client
        client = APIClient()

        # Create job specification
        job_spec = LeptonJobUserSpec()

        # Construct the original launch_processes command
        with open(args.config, "r") as f:
            config_content = f.read()
        launch_cmd = f"""\
cd {COSMOS_REASON1_DIR}

cat >config.toml <<EOF
{config_content}
EOF

python {TOOLS_RELATIVE_DIR}/launch_all.py --config config.toml"""

        # Get all non-Lepton arguments
        non_lepton_args = []
        for action in parser._actions:
            if hasattr(action, "option_strings") and action.option_strings:
                # Skip help action, lepton related arguments, and worker-idx
                if (
                    action.dest == "help"
                    or any(
                        opt.startswith("--lepton-") or opt == "--lepton-mode"
                        for opt in action.option_strings
                    )
                    or action.dest == "worker_idx"
                    or action.dest == "config"
                ):  # skip worker-idx
                    continue

                value = getattr(args, action.dest)
                if value is not None:
                    if isinstance(value, bool):
                        if value:
                            non_lepton_args.append(action.option_strings[0])
                    else:
                        non_lepton_args.append(f"{action.option_strings[0]} {value}")

        # Add all non-Lepton arguments to the command
        launch_cmd += " " + " ".join(non_lepton_args)

        # Handle node groups and queue priority
        if args.lepton_node_group or args.lepton_queue_priority:
            if args.lepton_queue_priority and not args.lepton_node_group:
                logger.error(
                    "Error: Queue priority is only available for dedicated node groups"
                )
                logger.error(
                    "Please use --lepton-queue-priority with --lepton-node-group"
                )
                sys.exit(1)

            node_group_ids = _get_valid_nodegroup_ids(
                args.lepton_node_group,
                need_queue_priority=(args.lepton_queue_priority is not None),
            )
            valid_node_ids = (
                _get_valid_node_ids(node_group_ids, args.lepton_node_id)
                if args.lepton_node_id
                else None
            )

            job_spec.affinity = LeptonResourceAffinity(
                allowed_dedicated_node_groups=node_group_ids,
                allowed_nodes_in_node_group=valid_node_ids,
            )

            if args.lepton_queue_priority:
                job_spec.queue_config = QueueConfig(
                    priority_class=args.lepton_queue_priority
                )

        # Set resource shape
        if args.lepton_resource_shape:
            job_spec.resource_shape = args.lepton_resource_shape
        else:
            available_types = "\n      ".join(VALID_SHAPES)
            logger.error(
                "Error: Missing option '--lepton-resource-shape'.\n"
                f"Available types are:\n      {available_types}.\n"
            )
            sys.exit(1)

        # Only for calculating the number of nodes needed.
        # Use this to replace the args.num_workers to calculate the number of nodes needed rather than specifying it using --num-workers
        # example

        match = re.search(r'(8|4|2)x', args.lepton_resource_shape)
        if match:
            num_gpus_per_node = int(match.group(1))
        else:
            num_gpus_per_node = 1
        global_launch_settings = replica_placement(
            list(range(num_gpus_per_node)),
            n_policy,
            n_rollouts,
            min_n_gpus_policy,
            min_n_gpus_rollout,
            "",
            "",
            None
        )
        if args.num_workers is not None:
            assert args.num_workers >= len(global_launch_settings)
        num_workers = len(global_launch_settings)
        logger.info(f"Number of workers required: {num_workers}")

        # Handle workers and communication
        if num_workers > 0:
            job_spec.completions = num_workers
            job_spec.parallelism = num_workers
            job_spec.intra_job_communication = True
        elif args.lepton_intra_job_communication is not None:
            job_spec.intra_job_communication = args.lepton_intra_job_communication

        # Set failure retry settings
        if args.lepton_max_failure_retry:
            job_spec.max_failure_retry = args.lepton_max_failure_retry
        if args.lepton_max_job_failure_retry:
            job_spec.max_job_failure_retry = args.lepton_max_job_failure_retry

        # Handle command
        job_spec.container.command = ["/bin/bash", "-c", launch_cmd]

        # Set container image
        if args.lepton_container_image:
            job_spec.container.image = args.lepton_container_image
        else:
            job_spec.container.image = BASE_IMAGE

        # Handle ports
        if args.lepton_container_port:
            job_spec.container.ports = [
                make_container_port_from_string(p) for p in args.lepton_container_port
            ]

        # Handle environment variables and secrets
        if args.lepton_env or args.lepton_secret:
            job_spec.envs = make_env_vars_from_strings(
                args.lepton_env, args.lepton_secret
            )

        # Handle mounts
        if args.lepton_mount:
            job_spec.mounts = make_mounts_from_strings(args.lepton_mount)

        # Set other configurations
        if args.lepton_image_pull_secrets:
            job_spec.image_pull_secrets = args.lepton_image_pull_secrets
        if args.lepton_privileged:
            job_spec.privileged = args.lepton_privileged
        if args.lepton_ttl_seconds_after_finished:
            job_spec.ttl_seconds_after_finished = args.lepton_ttl_seconds_after_finished
        if args.lepton_log_collection is not None:
            job_spec.log = LeptonLog(enable_collection=args.lepton_log_collection)
        if args.lepton_shared_memory_size is not None:
            job_spec.shared_memory_size = args.lepton_shared_memory_size

        # Handle reservation
        if args.lepton_with_reservation:
            if not args.lepton_node_group:
                logger.error(
                    "Error: --lepton-with-reservation is only supported for dedicated node groups"
                )
                sys.exit(1)
            job_spec.reservation_config = ReservationConfig(
                reservation_id=args.lepton_with_reservation
            )

        # Create job
        job = LeptonJob(
            spec=job_spec,
            metadata=Metadata(
                id=args.lepton_job_name,
                visibility=LeptonVisibility(args.lepton_visibility)
                if args.lepton_visibility
                else None,
            ),
        )

        # Create the job
        created_job = client.job.create(job)
        new_job_id = created_job.metadata.id_
        logger.info("🎉 Job Created Successfully!")
        logger.info(f"Name: {args.lepton_job_name}")
        logger.info(f"ID: {new_job_id}")

        return

    import cosmos_reason1.utils.util as util

    logger.info(f"Number of policy replicas: {n_policy} with {min_n_gpus_policy} gpus each")
    logger.info(f"Number of rollout replicas: {n_rollouts} with {min_n_gpus_rollout} gpus each")

    # Get available GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        raise RuntimeError("No GPUs available. Please check your GPU configuration.")


    # List of bash scripts to run (these should exist in the same directory)
    script_names = ["launch_controller.sh", "launch_replica.sh"]

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Verify scripts exist and are executable
    for script_name in script_names:
        script_path = os.path.join(script_dir, script_name)
        if not os.path.exists(script_path):
            logger.error(f"Error: Script {script_path} does not exist")
            sys.exit(1)
        if not os.access(script_path, os.X_OK):
            logger.error(f"Error: Script {script_path} is not executable")
            sys.exit(1)

    controller_script = os.path.join(script_dir, "launch_controller.sh")
    replica_script = os.path.join(script_dir, "launch_replica.sh")

    # Create commands for controller
    if args.log_dir is not None:
        output_dir = args.log_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(output_dir, f"logs_{timestamp}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = None

    control_url = None
    if args.url is not None:
        ip, port = args.url.split(":")
        if ip in get_local_ip():
            # If the IP is the local IP, launch the controller on the local machine
            port = util.find_available_port(int(port))
            logger.info(f"Using local IP: {ip} so launching controller on port {port}")
        else:
            control_url = args.url
    else:
        if (
            "LEPTON_JOB_WORKER_INDEX" in os.environ
            and int(os.environ.get("LEPTON_JOB_WORKER_INDEX")) != 0
        ):
            # For non-primary workers, connect to the primary worker (index 0) using its hostname
            prefix = os.environ.get(
                "LEPTON_JOB_SERVICE_PREFIX", os.environ.get("LEPTON_JOB_NAME")
            )
            subdomain = os.environ.get("LEPTON_SUBDOMAIN", "")
            primary_hostname = f"{prefix}-0.{subdomain}"
            control_url = f"{primary_hostname}:{args.port}"
        elif "LEPTON_JOB_WORKER_INDEX" in os.environ:
            # If we're in a Lepton job prime node, check if the port is available
            if not util.is_port_free(args.port):
                raise RuntimeError(f"Port {args.port} is not available")
            else:
                port = args.port
        else:
            port = util.find_available_port(args.port)

    controller_cmd = None
    tmpfile_toml = None
    if control_url is None:
        if n_policy > 0 or n_rollouts > 0:
            # Do not update the config if no replicas are needed which means launch controller only.
            if "policy" in cosmos_config and "parallelism" in cosmos_config["policy"]:
                cosmos_config["policy"]["parallelism"]["n_init_replicas"] = n_policy
            if "rollout" in cosmos_config and "parallelism" in cosmos_config["rollout"]:
                # Only available for RL.
                cosmos_config["rollout"]["parallelism"]["n_init_replicas"] = n_rollouts
        # Create a temporary file and write to it
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".toml", delete=False) as tmpfile:
            toml.dump(cosmos_config, tmpfile)
            tmpfile_toml = tmpfile.name
        logger.info(f"Temporary configuration file created at {tmpfile_toml}")
        controller_cmd = f"{controller_script} --config {tmpfile_toml}"
        controller_cmd += f" --port {port}"
        control_url = f"localhost:{port}"
    else:
        if "cosmos" in cosmos_config.get("train", {}).get("train_policy", {}).get("dataset_name", "").lower():
            logger.info("Prepare data for Lepton job, please wait...")
            from cosmos_reason1.policy.config import Config as PolicyConfig
            policy_config = PolicyConfig.from_dict(cosmos_config)
            from cosmos_reason1.utils.modelscope import update_config_if_modelscope
            policy_config = update_config_if_modelscope(policy_config)
            util.prepare_cosmos_data(config=policy_config)

    def get_lepton_ip(worker_idx: int) -> str:
        if "LEPTON_JOB_WORKER_INDEX" in os.environ:
            # For non-primary workers, connect to the primary worker (index 0) using its hostname
            prefix = os.environ.get(
                "LEPTON_JOB_SERVICE_PREFIX", os.environ.get("LEPTON_JOB_NAME")
            )
            subdomain = os.environ.get("LEPTON_SUBDOMAIN", "")
            hostname = f"{prefix}-{worker_idx}.{subdomain}"
        else:
            raise RuntimeError(
                "Lepton job worker index not found in environment variables"
            )
        return hostname

    def get_ip_from_list(worker_idx: int) -> str:
        if args.node_ip_list is not None:
            logger.info(f"Node IP list provided: {args.node_ip_list}")
            ip_list = args.node_ip_list.split(";")
            logger.info(f"Node IP list: {ip_list}")
            if worker_idx < len(ip_list):
                return ip_list[worker_idx]
            else:
                raise RuntimeError(
                    f"Worker index {worker_idx} exceeds the length of the IP list"
                )
        else:
            raise RuntimeError("Node IP list not provided")

    def get_worker_ip(worker_idx: int) -> str:
        if "LEPTON_JOB_WORKER_INDEX" in os.environ:
            return get_lepton_ip(worker_idx)
        elif args.node_ip_list is not None:
            return get_ip_from_list(worker_idx)
        else:
            return get_local_ip()[0]

    global_launch_settings = replica_placement(
        available_gpus,
        n_policy,
        n_rollouts,
        min_n_gpus_policy,
        min_n_gpus_rollout,
        replica_script,
        control_url,
        output_dir,
        get_worker_ip=get_worker_ip,
        rdzv_port=args.rdzv_port,
    )

    num_workers = len(global_launch_settings)
    logger.info(f"Number of workers required: {num_workers}")
    assert (
        len(available_gpus) * num_workers
        >= min_n_gpus_policy * n_policy + min_n_gpus_rollout * n_rollouts
    ), f"Not enough GPUs available. Required: {min_n_gpus_policy * n_policy + min_n_gpus_rollout * n_rollouts}, Available: {len(available_gpus)}"


    if (
        "LEPTON_JOB_WORKER_INDEX" in os.environ
        and int(os.environ.get("LEPTON_JOB_WORKER_INDEX")) >= 0
    ):
        cur_work_idx = int(os.environ.get("LEPTON_JOB_WORKER_INDEX"))
    else:
        cur_work_idx = args.worker_idx

    if len(global_launch_settings) <= cur_work_idx or len(global_launch_settings[cur_work_idx]) == 0:
        if controller_cmd is None:
            logger.info(
                f"No launch settings found for worker index {cur_work_idx}, no need launch"
            )
            sys.exit(0)

    processes = []

    controller_id = -1

    if controller_cmd is not None:
        controller_process = launch_processes(
            [controller_cmd],
            [""],
            [""],
            [os.path.join(output_dir, "controller.log") if output_dir is not None else None],
        )
        controller_id = len(processes)
        processes.append(controller_process[0])

    logger.info(f"Waiting for controller to be ready at {control_url}")
    wait_for_url_ready(control_url)
    logger.info(f"Controller is ready at {control_url}")


    if len(global_launch_settings) > cur_work_idx and len(global_launch_settings[cur_work_idx]) != 0:
        commands = global_launch_settings[cur_work_idx][0]
        gpu_devices = global_launch_settings[cur_work_idx][1]
        control_urls = global_launch_settings[cur_work_idx][2]
        output_files = global_launch_settings[cur_work_idx][3]

        # Combine all commands
        logger.info(f"Commands to be executed: {commands}")
        logger.info(f"GPU devices to be used: {gpu_devices}")
        logger.info(f"Control URLs to be used: {control_urls}")
        logger.info(f"Output files: {output_files}")

        # Check if the number of GPU devices matches the number of commands
        assert (
            len(gpu_devices) == len(commands)
        ), f"Number of GPU devices ({len(gpu_devices)}) does not match number of commands ({len(commands)})"

        # Launch all processes
        processes.extend(launch_processes(commands, gpu_devices, control_urls, output_files))

    if tmpfile_toml is not None and os.path.exists(tmpfile_toml):
        # Clean up the temporary file
        try:
            os.unlink(tmpfile_toml)
        except Exception as e:
            logger.error(f"Error deleting temporary file {tmpfile_toml}: {e}")

    # Wait for all processes to complete without blocking
    while len(processes) > 0:
        for i, process in enumerate(processes):
            try:
                # Check if process has finished without blocking
                if process.poll() is not None:
                    returncode = process.returncode
                    if returncode == 0:
                        logger.info(f"Process {i} completed successfully")
                    else:
                        logger.error(f"Process {i} failed with return code {returncode}")
                        # Terminate all remaining processes
                        if controller_id == -1 or i == controller_id:
                            for p in processes:
                                try:
                                    p.kill()
                                except Exception as e:
                                    logger.error(f"Error kill process {p}: {e}")
                            logger.error("Terminated all processes due to failure")
                            sys.exit(1)  # Exit with error code 1 if any process failed
                    # Remove completed process from list
                    processes.remove(process)
            except Exception as e:
                logger.error(f"Error monitoring process {i}: {e}")
                # Terminate all remaining processes
                if controller_id == -1 or i == controller_id:
                    for p in processes:
                        try:
                            p.kill()
                        except Exception as e:
                            logger.error(f"Error kill process {p}: {e}")
                    logger.error("Terminated all processes due to error")
                    sys.exit(1)
        # Small sleep to prevent busy waiting
        time.sleep(0.1)


if __name__ == "__main__":
    main()
