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

import logging
from typing import List, Literal
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util import NodeLaunchMetadata, ReplicaLaunchMetadata
import argparse
import math
import tempfile
import subprocess
import json
import toml

logging.basicConfig(level=logging.INFO)

def compute_nodes(n_gpu_per_node: int, n_gpu_per_replica: int, n_replicas: int, role: Literal["policy", "rollout"]) -> List[NodeLaunchMetadata]:
    '''
    Compute the number of nodes required for a given number of GPUs per replica and the number of replicas.

    If multiple replicas are colocated on the same node, the visible GPUs for each replica are computed.
    
    Returns:
        A list of NodeLaunchMetadata, one for each node.
    '''
    n_nodes = 0
    rendezvous_port = 29345

    node_launch_metadata = []
    if n_gpu_per_replica >= n_gpu_per_node:
        assert n_gpu_per_replica % n_gpu_per_node == 0, f"Number of GPUs per policy must be a multiple of {n_gpu_per_node}"
        n_policy_nodes = n_replicas * (n_gpu_per_replica // n_gpu_per_node)

        rendezvous_node = 0
        for i_node in range(n_policy_nodes):
            if i_node % (n_gpu_per_replica // n_gpu_per_node) == 0:
                rendezvous_node = i_node
            
            replica_launch_meta = [
                # Only one replica per node, no colocation or rendezvous conflicts
                ReplicaLaunchMetadata(
                    nnode=n_gpu_per_replica // n_gpu_per_node,
                    role=role,
                    rendezvous_node=rendezvous_node,
                    rendezvous_port=rendezvous_port,
                    visible_gpus=list(range(0, n_gpu_per_node))
                )
            ]
            node_launch_metadata.append(NodeLaunchMetadata(colocation=replica_launch_meta))
    else:
        possible_gpu_per_replica = []
        for divisor in range(1, n_gpu_per_node):
            if n_gpu_per_node % divisor == 0:
                possible_gpu_per_replica.append(divisor)

        assert n_gpu_per_replica in possible_gpu_per_replica, f"Number of GPUs per policy must be one of {possible_gpu_per_replica}, got {n_gpu_per_replica}."
        n_policy_nodes = math.ceil(n_replicas * n_gpu_per_replica / n_gpu_per_node)

        replica_counter = 0
        for i_node in range(n_policy_nodes):
            replica_launch_meta = []
            local_replica_counter = 0
            while replica_counter < n_replicas:
                replica_launch_meta.append(ReplicaLaunchMetadata(
                    nnode=1,
                    role=role,
                    rendezvous_node=i_node, # Always on the same node
                    rendezvous_port=rendezvous_port + replica_counter, # To avoid conflicts with other replicas on the same node
                    visible_gpus=list(range(local_replica_counter * n_gpu_per_replica, (local_replica_counter + 1) * n_gpu_per_replica))
                ))
                replica_counter += 1
                local_replica_counter += 1
                if replica_counter == n_replicas:
                    break
                elif local_replica_counter * n_gpu_per_replica >= n_gpu_per_node:
                    # Dispatch left to next node
                    break
            node_launch_metadata.append(NodeLaunchMetadata(colocation=replica_launch_meta))
    n_nodes += n_policy_nodes

    return node_launch_metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", type=str, default="cosmos_job")
    parser.add_argument("--ngpu-per-node", type=int, default=8, help="Number of GPUs per compute node.")
    parser.add_argument("--n-policy-replicas", type=int, default=1, help="Number of policy replicas to launch")
    parser.add_argument("--n-rollout-replicas", type=int, default=1, help="Number of rollout replicas to launch")
    parser.add_argument("--slurm-partition", type=str, default="batch", help="SLURM partition to use")
    parser.add_argument("--slurm-account", type=str, default="sw_aidot", help="SLURM account to use")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the controller config file")
    parser.add_argument("--repo-root-path", type=str, required=True, help="Path to the repository root")
    parser.add_argument("--output-root-path", type=str, required=True, help="Path to the output root")
    parser.add_argument("--cosmos-container", type=str, required=True, help="Path to the cosmos container")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = toml.load(f)
    min_n_gpus_policy = (
        config['policy']['parallelism']['tp_size']
        * config['policy']['parallelism']['dp_replicate_size']
        * config['policy']['parallelism']['pp_size']
        * config['policy']['parallelism']['cp_size']
    )
    min_n_gpus_rollout = (
        config['rollout']['parallelism']['tp_size']
        * config['rollout']['parallelism']['dp_replicate_size']
        * config['rollout']['parallelism']['pp_size']
        * config['rollout']['parallelism']['cp_size']
    )
    if config['policy']['parallelism']['dp_shard_size'] >= 1:
        min_n_gpus_policy = (
            min_n_gpus_policy * config['policy']['parallelism']['dp_shard_size']
        )
    if config['rollout']['parallelism']['dp_shard_size'] >= 1:
        min_n_gpus_rollout = (
            min_n_gpus_rollout * config['rollout']['parallelism']['dp_shard_size']
        )

    policy_node_launch_metadata: List[NodeLaunchMetadata] = compute_nodes(args.ngpu_per_node, min_n_gpus_policy, args.n_policy_replicas, "policy")
    rollout_node_launch_metadata: List[NodeLaunchMetadata] = compute_nodes(args.ngpu_per_node, min_n_gpus_rollout, args.n_rollout_replicas, "rollout")
    
    n_policy_nodes = len(policy_node_launch_metadata)
    n_rollout_nodes = len(rollout_node_launch_metadata)

    # Template for the slurm script
    template_vars = {
        "TOTAL_NODES": f"{n_policy_nodes + n_rollout_nodes}",
        "REPO_ROOT_PATH": args.repo_root_path,
        "OUTPUT_ROOT_PATH": args.output_root_path,
        "COSMOS_CONTAINER": args.cosmos_container,
        "SLURM_PARTITION": args.slurm_partition,
        "SLURM_ACCOUNT": args.slurm_account,
        "SLURM_JOB_NAME": args.job_name,
        "CONFIG_PATH": args.config_path,
    }

    # Environment variables
    env_vars = {
        "NUM_POLICY_NODES": f"{n_policy_nodes}",
        "NUM_ROLLOUT_NODES": f"{n_rollout_nodes}",
        "TOTAL_NODES": f"{n_policy_nodes + n_rollout_nodes}",
    }
    
    # Read the template relative to the current file
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cosmos_reason1_job_multi_node.sh"), "r") as f:
        template = f.read()
    
    # Replace the template variables
    for key, value in template_vars.items():
        template = template.replace(f"[[{key}]]", value)

    # Write the template to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(template.encode("utf-8"))
    temp_file.close()
    logging.info(f"Template written to {temp_file.name}")

    env = os.environ.copy()
    env.update(env_vars)
    env["NODE_LAUNCH_METADATA_POLICY"] = json.dumps([x.to_json() for x in policy_node_launch_metadata])
    env["NODE_LAUNCH_METADATA_ROLLOUT"] = json.dumps([x.to_json() for x in rollout_node_launch_metadata])
    proc = subprocess.Popen(["sbatch", temp_file.name], env=env)
    proc.wait()
    if proc.returncode != 0:
        logging.error(f"Failed to submit job: {proc.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
