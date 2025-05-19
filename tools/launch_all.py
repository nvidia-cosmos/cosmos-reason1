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
import time
import os
import argparse
from typing import List, Dict, Optional, Any
from cosmos_reason1.utils.logging import logger


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
    import tomli

    try:
        with open(config_file, "rb") as f:
            config = tomli.load(f)
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
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save logs. If not provided, logs will be printed to stdout.",
    )
    parser.add_argument(
        "-wc",
        "--weight-sync-check",
        action="store_true",
        default=False,
        help="Whether to do weight sync correctness check between policy and rollout for debugging.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers to use for the job, default is 1. This is used when multi-node training are used for the job.",
    )
    parser.add_argument(
        "--worker-idx",
        type=int,
        default=0,
        help="Worker index for local execution. In Lepton mode, this is ignored as worker indices are automatically assigned by Lepton.",
    )

    parser.add_argument(
        "--lepton-mode",
        action="store_true",
        default=False,
        help="Enable Lepton mode for remote execution",
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
        return local_ip
    except Exception as e:
        logger.error(f"Error getting local IP address: {e}")
        return None


def main():
    args, parser = parse_args()
    if args.weight_sync_check:
        # This launch is only for weight sync correctness check
        os.environ["COSMOS_WEIGHT_SYNC_CHECK"] = "true"

    # Handle Lepton mode
    if args.lepton_mode:
        from leptonai.api.v1.client import APIClient
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
        launch_cmd = f"cd {COSMOS_REASON1_DIR} && python {TOOLS_RELATIVE_DIR}/launch_all.py"

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

        # Handle workers and communication
        if args.num_workers:
            job_spec.completions = args.num_workers
            job_spec.parallelism = args.num_workers
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

    from cosmos_reason1.policy.config import Config as PolicyConfig
    import cosmos_reason1.utils.util as util

    # Check if the config file is provided
    cosmos_config = PolicyConfig.from_dict(read_config(args.config))
    # Get the number of GPUs required for policy and rollout
    # and the number of replicas for each
    if args.policy is None:
        n_policy = cosmos_config.policy.parallelism.n_init_replicas
    else:
        n_policy = args.policy
    if args.rollout is None:
        n_rollouts = cosmos_config.rollout.parallelism.n_init_replicas
    else:
        n_rollouts = args.rollout

    # If the training type is SFT, set n_rollouts to 0
    if cosmos_config.train.train_policy.type == "sft":
        n_rollouts = 0

    logger.info(f"Number of policy replicas: {n_policy}")
    logger.info(f"Number of rollout replicas: {n_rollouts}")
    # Calculate the minimum number of GPUs required for policy and rollout
    # based on the parallelism settings in the configuration
    # Treat dp_shard_size as 1 if it is not set
    min_n_gpus_policy = (
        cosmos_config.policy.parallelism.tp_size
        * cosmos_config.policy.parallelism.dp_replicate_size
        * cosmos_config.policy.parallelism.pp_size
        * cosmos_config.policy.parallelism.cp_size
    )
    min_n_gpus_rollout = (
        cosmos_config.rollout.parallelism.tp_size
        * cosmos_config.rollout.parallelism.dp_replicate_size
        * cosmos_config.rollout.parallelism.pp_size
        * cosmos_config.rollout.parallelism.cp_size
    )
    if cosmos_config.policy.parallelism.dp_shard_size >= 1:
        min_n_gpus_policy = (
            min_n_gpus_policy * cosmos_config.policy.parallelism.dp_shard_size
        )
    if cosmos_config.rollout.parallelism.dp_shard_size >= 1:
        min_n_gpus_rollout = (
            min_n_gpus_rollout * cosmos_config.rollout.parallelism.dp_shard_size
        )

    num_workers = args.num_workers

    # Get available GPUs
    available_gpus = get_available_gpus()
    assert (
        len(available_gpus) * num_workers
        >= min_n_gpus_policy * n_policy + min_n_gpus_rollout * n_rollouts
    ), f"Not enough GPUs available. Required: {min_n_gpus_policy * n_policy + min_n_gpus_rollout * n_rollouts}, Available: {len(available_gpus)}"
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
    commands = []
    gpu_devices = []
    control_urls = []
    output_files = []
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
        if ip == get_local_ip():
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
    if control_url is None:
        controller_cmd = f"{controller_script} --config {args.config}"
        controller_cmd += f" --port {port}"
        control_url = f"localhost:{port}"
    else:
        if "cosmos" in cosmos_config.train.train_policy.dataset_name.lower():
            logger.info("Prepare data for Lepton job, please wait...")
            util.prepare_cosmos_data(config=cosmos_config)

    # Prepare the command to launch the controller for all workers
    global_available_gpus = [available_gpus for _ in range(num_workers)]
    # Create commands for policy and rollout replicas
    gpu_idx = 0
    global_worker_idx = 0
    global_launch_settings = [[] for _ in range(num_workers)]

    # Assign launch settings for each worker
    for i in range(n_policy):
        if gpu_idx + min_n_gpus_policy > len(global_available_gpus[global_worker_idx]):
            global_launch_settings[global_worker_idx].extend(
                [commands, gpu_devices, control_urls, output_files]
            )
            commands = []
            gpu_devices = []
            control_urls = []
            output_files = []
            gpu_idx = 0
            global_worker_idx += 1

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

    for i in range(n_rollouts):
        if gpu_idx + min_n_gpus_rollout > len(global_available_gpus[global_worker_idx]):
            global_launch_settings[global_worker_idx].extend(
                [commands, gpu_devices, control_urls, output_files]
            )
            commands = []
            gpu_devices = []
            control_urls = []
            output_files = []
            gpu_idx = 0
            global_worker_idx += 1

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
        global_launch_settings[global_worker_idx].extend(
            [commands, gpu_devices, control_urls, output_files]
        )

    if (
        "LEPTON_JOB_WORKER_INDEX" in os.environ
        and int(os.environ.get("LEPTON_JOB_WORKER_INDEX")) >= 0
    ):
        cur_work_idx = int(os.environ.get("LEPTON_JOB_WORKER_INDEX"))
    else:
        cur_work_idx = args.worker_idx

    if len(global_launch_settings[cur_work_idx]) == 0:
        logger.info(
            f"No launch settings found for worker index {cur_work_idx}, no need launch"
        )
        sys.exit(0)

    commands = global_launch_settings[cur_work_idx][0]
    gpu_devices = global_launch_settings[cur_work_idx][1]
    control_urls = global_launch_settings[cur_work_idx][2]
    output_files = global_launch_settings[cur_work_idx][3]

    processes = []

    if controller_cmd is not None:
        controller_process = launch_processes(
            [controller_cmd],
            [""],
            [""],
            [os.path.join(output_dir, "controller.log") if output_dir is not None else None],
        )
        processes.append(controller_process[0])

    logger.info(f"Waiting for controller to be ready at {control_url}")
    wait_for_url_ready(control_url)
    logger.info(f"Controller is ready at {control_url}")

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

    # Wait for all processes to complete
    for i, process in enumerate(processes, 1):
        try:
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                logger.info(f"Process {i} completed successfully")
                if stdout is not None:
                    logger.info(f"Output: {stdout.strip()}")
            else:
                logger.info(f"Process {i} failed with return code {process.returncode}")
                if stderr is not None:
                    logger.info(f"Error: {stderr.strip()}")
        except Exception as e:
            logger.error(f"Error waiting for process {i}: {e}")


if __name__ == "__main__":
    main()
