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
import sys
import glob
import toml
import atexit
import subprocess
import torch

WORK_DIR = "/tmp/test_grad_allreduce"
CTRL_URL = ""


def write_train_config(dp_replica: int = 1, dp_shard: int = 1):
    # make sure every step has the same batch size
    train_batch_per_replica = 8 // dp_replica

    config = f"""
redis = "12808"

[train]
resume = false
epoch = 1
output_dir = "{WORK_DIR}"
epsilon = 1e-6
optm_name = "AdamW"
optm_lr = 1e-6
optm_impl = "fused"
optm_weight_decay = 0.01
optm_betas = [ 0.9, 0.999,]
optm_warmup_steps = 20
optm_grad_norm_clip = 1.0
async_tp_enabled = false
compile = false
param_dtype = "bfloat16"
fsdp_reduce_dtype = "float32"
fsdp_offload = false
fsdp_reshard_after_forward = "default"
train_batch_per_replica = {train_batch_per_replica}
sync_weight_interval = 1

[rollout]
gpu_memory_utilization = 0.7
enable_chunked_prefill = false
max_response_length = 2048
n_generation = 16
batch_size = 1
quantization = "none"


[policy]
model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
model_max_length = 4096
model_gradient_checkpointing = true

[logging]
logger = ['console', 'wandb']
project_name = "cosmos_reason1"
experiment_name = "None"

[train.train_policy]
type = "grpo"
dataset.name = "JiaxinTsao/math_examples"
prompt_column_name = "prompt"
response_column_name = "result"
reward_function = "boxed_math"
dataset.train_split="train"
temperature = 1
epsilon_low = 0.2
epsilon_high = 0.2
kl_beta = 0.0

[train.ckpt]
enable_checkpoint = true
save_freq = 20
save_mode = "async"

[rollout.parallelism]
n_init_replicas = 1
tp_size = 2
pp_size = 1

[rollout.sampling_config]
temperature = 0.9
top_p = 1.0
top_k = 10

[policy.parallelism]
n_init_replicas = 1
tp_size = 2
cp_size = 1
dp_shard_size = {dp_shard}
pp_size = 1
dp_replicate_size = 1
cp_rotate_method = "allgather"
"""
    cfg_file_path = os.path.join(WORK_DIR, "train_config.toml")
    with open(cfg_file_path, "w") as f:
        f.write(config)
    return cfg_file_path


def launch_controller(config: str, port: int = 8010):
    """
    Launch the controller process.
    """
    print("launch controller")
    p = subprocess.Popen(
        "python -m cosmos_reason1.dispatcher.run_web_panel "
        f"--port {port} --config {config}",
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        env=dict(os.environ),
    )

    # get the controller listen port
    os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{port}"
    global CTRL_URL
    CTRL_URL = os.environ["COSMOS_CONTROLLER_HOST"]
    return [p]


# launch_rollout will send training data to the controller in reproduciable sort
def launch_rollout(config: str):
    """
    Launch the rollout process.
    """
    print("launch rollout")

    cmd = f"CUDA_VISIBLE_DEVICES=6,7 COSMOS_CONTROLLER_HOST={CTRL_URL} ./tools/launch_replica.sh --type rollout --ngpus 2"
    print(f"launch rollout with cmd: {cmd}")
    return [
        subprocess.Popen(
            cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True
        )
    ]


def launch_policy(config: str, gpus: str):
    """
    Launch the policy process.
    """
    print("launch policy")
    cfg = toml.load(config)
    policy_p = cfg["policy"]["parallelism"]
    ngpu = (
        int(policy_p["dp_shard_size"])
        * int(policy_p["tp_size"])
        * int(policy_p["pp_size"])
    )
    assert ngpu <= 4, "for test, policy will max use 4 gpus"

    cmd = f"CUDA_VISIBLE_DEVICES={gpus} COSMOS_CONTROLLER_HOST={CTRL_URL} ./tools/launch_replica.sh --type policy --ngpus {ngpu}"
    print(f"launch policy with cmd: {cmd}")
    return [
        subprocess.Popen(
            cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True
        )
    ]


def compare_ckpt_grad(dp_shard_ckpt_path: str, dp_replica_ckpt_path: str):
    """
    Compare the gradients of the checkpointed model with the original model.
    """
    dp_shard_ckpt = torch.load(dp_shard_ckpt_path)
    dp_replica_ckpt = torch.load(dp_replica_ckpt_path)

    print("=========== Gradient Compare result: ===========")
    print(f" {dp_shard_ckpt_path}\n {dp_replica_ckpt_path}")
    for key in dp_shard_ckpt.keys():
        if key not in dp_replica_ckpt:
            print(f"Key {key} not found in dp_replica_ckpt")
            continue
        # for dtype bfloat16, we release tolerance to 1e-3
        if not torch.allclose(
            dp_shard_ckpt[key], dp_replica_ckpt[key], atol=1e-3, rtol=1e-3
        ):
            print(f"Gradients do not match for key {key}")
            return False
    print("All gradients match!")
    return True


popens = []


def cleanup():
    """
    Cleanup function to kill all processes.
    """
    for popen in popens:
        popen.kill()
    popens.clear()

    # try kill created processes
    try:
        login_user = os.getlogin()
    except Exception:
        login_user = "root"

    subprocess.run(f"pkill -u {login_user} -f 'cosmos'", shell=True)
    subprocess.run(f"pkill -u {login_user} -f 'redis'", shell=True)
    subprocess.run(f"pkill -u {login_user} -f 'torchrun'", shell=True)


atexit.register(cleanup)


if __name__ == "__main__":
    # Set the environment variable for the test
    os.environ["COSMOS_TEST"] = "1"
    os.environ["COSMOS_LOG_LEVEL"] = "INFO"

    # Create the work directory
    os.makedirs(WORK_DIR, exist_ok=True)

    # # step 1: run a normal training with dp_replica=1, dp_shard=2
    # print("step 1: run a normal training with dp_replica=1, dp_shard=2")
    # # Write the training config file
    # config = write_train_config(dp_replica=1, dp_shard=2)
    # # Launch the controller
    # popens += launch_controller(config)
    # # Launch the rollout
    # popens += launch_rollout(config)
    # # Launch the policy process
    # popens += launch_policy(config, "0,1,2,3")
    # # wait for policy finish
    # popens[-1].wait()
    # # kill all processes
    # cleanup()

    # # wait for a while to ensure all processes are killed
    # time.sleep(2)

    # step 2: run a normal training with dp_replica=2, dp_shard=1
    print("step 2: run a normal training with dp_replica=2, dp_shard=1")
    # Write the training config file
    config = write_train_config(dp_replica=2, dp_shard=1)
    # Launch the controller
    popens += launch_controller(config, 8300)
    # Mock rollout
    popens += launch_rollout(config)
    # Launch 2 policy process
    popens += launch_policy(config, "0,1")
    popens += launch_policy(config, "2,3")

    # wait for policy finish
    popens[-1].wait()
    # kill all processes
    cleanup()

    # step 3: compare the gradients of the checkpointed model with the original model
    print(
        "step 3: compare the gradients of the checkpointed model with the original model"
    )
    dumped_grad = glob.glob(os.path.join(WORK_DIR, "grad_embed_tokens.weight-*.pt"))
    dp1r2 = [fn for fn in dumped_grad if "1-2" in os.path.basename(fn)][0]
    dp2r1 = [fn for fn in dumped_grad if "2-1" in os.path.basename(fn)][0]

    compare_ckpt_grad(
        dp1r2,
        dp2r1,
    )

    # step 4: clean file
    for fn in dumped_grad:
        os.remove(fn)
