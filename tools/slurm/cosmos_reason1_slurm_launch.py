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

'''
This script is launched by slurm job.
'''

import argparse
import logging
from typing import List
import subprocess, time
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util import NodeLaunchMetadata, ReplicaLaunchMetadata


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, choices=["policy", "rollout"])
    args = parser.parse_args()

    node_list = os.environ["LOCAL_NODE_LIST"].split(" ")
    COSMOS_CONTROLLER_HOST = os.environ["COSMOS_CONTROLLER_HOST"]
    self_node = os.environ["SLURMD_NODENAME"]
    node_launch_metadata: List[NodeLaunchMetadata] = NodeLaunchMetadata.from_json_list(os.environ[f"NODE_LAUNCH_METADATA_{args.type.upper()}"])

    # Either the policy or rollout nodes
    logging.info(f"COSMOS_CONTROLLER_HOST: {COSMOS_CONTROLLER_HOST}")
    logging.info(f"NODE LIST: {node_list}")
    assert self_node in node_list, f"self_node {self_node} not in node_list {node_list}"
    self_node_idx = node_list.index(self_node)
    self_node_launch_metadata = node_launch_metadata[self_node_idx]

    cmds = []
    envs = []
    for replica_launch_metadata in self_node_launch_metadata.colocation:
        rendezvous_node = node_list[replica_launch_metadata.rendezvous_node]
        rendezvous_port = replica_launch_metadata.rendezvous_port
        visible_gpus = replica_launch_metadata.visible_gpus
        nnode = replica_launch_metadata.nnode

        logging.info(f"Rendezvous node: {rendezvous_node}, rendezvous port: {rendezvous_port}, visible GPUs: {visible_gpus}")
        env  = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in visible_gpus)
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        replica_launch_script = os.path.join(os.path.dirname(__file__), "..", "launch_replica.sh")
        cmds.append([replica_launch_script, "--type", args.type, "--rdzv-endpoint", f"{rendezvous_node}:{rendezvous_port}", "--ngpus", str(len(visible_gpus)), "--nnodes", str(nnode)])
        envs.append(env)
    
    procs = [subprocess.Popen(cmd, env=env) for cmd, env in zip(cmds, envs)]
    # block until every process finishes, and propagate any non-zero exit codes
    exit_code = 0
    while len(procs) > 0:
        for i, p in enumerate(procs):
            try:
                # Check if process has finished without blocking
                if p.poll() is not None:
                    returncode = p.returncode
                    if returncode != 0:
                        # Kill and exit if any process fails
                        for p in procs:
                            try:
                                p.kill()
                            except Exception:
                                pass
                        sys.exit(returncode)
                    # Remove completed process from list
                    procs.remove(p)
            except Exception:
                # Terminate all remaining processes
                for p in procs:
                    try:
                        p.kill()
                    except Exception:
                        pass
                sys.exit(1)
        # Small sleep to prevent busy waiting
        time.sleep(0.1)
    sys.exit(exit_code)              # mimic “all-good” or the first failure


