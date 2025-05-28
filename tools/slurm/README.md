# Launch Job by Slurm

We create a script for easily launching training job by slurm on the cluster. For example, if you want to launch a job with `1` replica for policy, `2` replica for rollout, 8 GPUs for each policy, and 4 GPUs for each rollout, you can run the following commands:
```bash
python ./tools/slurm/dispatch_job.py \
    --ngpu-per-policy 8 \
    --ngpu-per-rollout 4 \
    --n-policy-replicas 1 \
    --n-rollout-replicas 2  \
    --config-path configs/qwen3/qwen3-32b-p-fsdp1-tp8-r-tp4-pp1-grpo.toml \
    --repo-root-path ${COSMOS_PATH} \
    --output-root-path ${COSMOS_OUTPUT_PATH} \
    --cosmos-container ${COSMOS_SQSQ_PATH}

### Output:
Submitted batch job 2878971
```
Total number of GPU node is computed and dispatched via slurm platform automatically.

The full arguments for `dispatch_job.py` are as follows:

| Argument               | Type                     | Required | Default Value | Description                                                                 |
|------------------------|--------------------------|----------|---------------|-----------------------------------------------------------------------------|
| --job-name             | str                      | No       | cosmos_job    | Name of the SLURM job.                                                      |
| --ngpu-per-node        | int                      | No       | 8             | Number of GPUs per compute node.                                            |
| --ngpu-per-policy      | int                      | Yes      |  -             | Number of GPUs per policy replica, must be compatible with parallelism config in the controller config file |
| --ngpu-per-rollout     | int                      | Yes      |  -             | Number of GPUs per rollout replica, must be compatible with parallelism config in the controller config file |
| --n-policy-replicas    | int                      | No       | 1             | Number of policy replicas to launch.                                        |
| --n-rollout-replicas   | int                      | No       | 1             | Number of rollout replicas to launch.                                       |
| --slurm-partition      | str                      | No       | batch         | SLURM partition to use.                                                     |
| --slurm-account        | str                      | No       | sw_aidot      | SLURM account to use.                                                       |
| --config-path          | str                      | Yes      |   -            | Path to the controller config file.                                         |
| --repo-root-path       | str                      | Yes      |   -            | Path to the repository root.                                                |
| --output-root-path     | str                      | Yes      |   -            | Path to the output root.                                                    |
| --cosmos-container     | str                      | Yes      |   -            | Path to the cosmos container.                                               |