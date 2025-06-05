## Profiling the training in cosmos

We support profiling the training in cosmos by separate ranks. Users must specify the profiling config in `toml` config file like:

```
[profiler]
enable_profiler = true

[profiler.sub_profiler_config]
active_steps = 2
rank_filter = [0]
record_shape = false
profile_memory = false
with_stack = false
with_modules = false

```

| Attribute         | Detail                                                                                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `enable_profiler` | Enable profiler for training, `false` by default.                                                                                                            |
| `active steps`    | Number of Steps that the profiler will trace the perf. By default, the profiler will skip the first two steps as wait and warmup, then trace `active_steps`. |
| `rank_filter`     | Filter of ranks. Only ranks in this list will actually do the profiling.                                                                                     |
| `record_shape`    | Same in torch profiler                                                                                                                                       |
| `profile_memory`  | Same in torch profiler                                                                                                                                       |
| `with_stack`      | Same in torch profiler                                                                                                                                       |
| `profile_memory`  | Same in torch profiler                                                                                                                                       |



### How to use profiler

#### 1. Enable the profiling

Set the `enable_profiler` flag to `true` in the config file. Carefully set the `active_steps` because the output trace file could be very huge.

#### 2. Launch the RL

Launch Controller, Policy, and Rollout wrokers like normal.

#### 3. Send profile command to controller

Because we are in async mode, by default the profiling won't start automatically to avoid non-sense tracing. Users could send a command to the controller to specify which replica will start the profiling.

Example:
```
cosmos profile set -ch localhost -cp 8000 a8563b86-5d7a-4955-82f1-e8e1b5d7c6f3
```

Output:
```
Set replica a8563b86-5d7a-4955-82f1-e8e1b5d7c6f3 to profile mode.
```

Then the controller will print logs like:

```
[cosmos] 2025-05-29 04:52:43,028 - cosmos - INFO - [Controller] Set profile for replica a8563b86-5d7a-4955-82f1-e8e1b5d7c6f3.
```

This means the controller will tell the specified policy replica to start doing profiling at the coming round of training step. Remember that this command will only take effect for once for the same replica. Sending multiple commands to the same replica will generate multiple rounds of trace file.

User could also specify profiler args dynamically, please run `cosmos profile set --help` to get the details.

#### 4. Get the trace file.

By default, the trace file after profiling is finished will be stored to `${config.train.output_dir}/profile_trace/${REPLICA_NAME}_${GLOBAL_RANK}/${TRACE_COUNT}_trace.json.gz`.


If the user specifies the S3 storage config, this trace file will also be uploaded to the S3 storage.

The trace file will be attached to the atom/device that running the profiling. This info will be reported to controller after profiling.

### Visualize the trace file

Users could download the trace file, and open it with [Google perfetto](https://ui.perfetto.dev/) for visualization. 

Notes: the trace file could be huge and `perfetto` will fail, users could follow the [Visualising large traces](https://perfetto.dev/docs/visualization/large-traces) to optimize the trace file.

### Example

1. Launch rl with a config that enables profiling
the config file `qwen2-5-7b-p-fsdp1-tp2-r-tp2-pp1-grpo-profile.toml` has already been set to allow for profiling and will profile 1 training step on rank 0 of a policy replica.
```
python tools/launch_all.py --config ./configs/qwen2-5/qwen2-5-7b-p-fsdp1-tp2-r-tp2-pp1-grpo-profile.toml --debug
```
2. Specify which policy replica to run the profiling

```
cosmos profile set -ch localhost -cp 8000 a8563b86-5d7a-4955-82f1-e8e1b5d7c6f3
```

3. Get the trace file.

From the local disk, we could find the trace file: `outputs/qwen2-5-7b-p-fsdp1-tp2-r-tp2-pp1-grpo-profile/profile_trace/a8563b86-5d7a-4955-82f1-e8e1b5d7c6f3_0/0_trace.json.gz`.


### For SFT

SFT profiling is also supported. Unlike RL, the profiler will start automatically at the beginning of the training. And then it will profile like normal.

When active steps are reached, the profiler will stop and save the trace file.

For example:

```
python tools/launch_all.py --config ./configs/cosmos-reason1/cosmos-reason1-7b-tp2-sft-profile.toml --debug
```

Then after the active steps, the trace file will be saved to `outputs/cosmos-reason1-7b-tp2-sft-profile/profile_trace/4e52c790-ccc7-4a02-903f-65da9a02f9fd_0/0_trace.json.gz`.
