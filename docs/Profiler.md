## Profiling the training in cosmos

We support profiling the training in cosmos by separate ranks. Users must specify the profiling config in `toml` config file like:

```
[profiler]
enable_profiler = true
active_steps = 2
rank_filter = [0,]
```

| Attribute         | Detail                                                                                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `enable_profiler` | Enable profiler for training, `false` by default.                                                                                                            |
| `active steps`    | Number of Steps that the profiler will trace the perf. By default, the profiler will skip the first two steps as wait and warmup, then trace `active_steps`. |
| `rank_filter`     | Filter of ranks. Only ranks in this list will actually do the profiling.                                                                                     |


### How to use profiler

#### 1. Enable the profiling

Set the `enable_profiler` flag to `true` in the config file. Carefully set the `active_steps` because the output trace file could be very huge.
#### 2. Launch the RL

Launch Controller, Policy, and Rollout wrokers like normal.

#### 3. Send profile command to controller

Because we are in async mode, by default the profiling won't start automatically to avoid non-sense tracing. Users could send a command to the controller to specify which replica will start the profiling. We offer a simple script `tools/profile_cmd.py` to do this.

Example:

```
./tools/profile_cmd.py -c CONTROLLER_IP:PORT -r POLICY_REPLICA_NAME
```

Then the controller will print logs like:

```
[cosmos] 2025-05-29 04:52:43,028 - cosmos - INFO - [Controller] Set profile for replica d86e828c-5b84-42a1-8de4-b23e247a7352.
```

This means the controller will tell the specified policy replica to start doing profiling at the coming round of training step. Remember that this command will only take effect for once for the same replica. Sending multiple commands to the same replica won't generate multiple rounds of trace files.

#### 4. Get the trace file.

By default, the trace file after profiling is finished will be stored to `${config.train.output_dir}/profile_trace/${REPLICA_NAME}_${GLOBAL_RANK}/trace.json`.


If the user specifies the S3 storage config, this trace file will also be uploaded to the S3 storage.

The trace file will be attached to the atom/device that running the profiling. This info will be reported to controller after profiling.

### Visualize the trace file

Users could download the trace file, and open it with [Google perfetto](https://ui.perfetto.dev/) for visualization. 

Notes: the trace file could be huge and `perfetto` will fail, users could follow the [Visualising large traces](https://perfetto.dev/docs/visualization/large-traces) to optimize the trace file.

### Example

1. Launch rl with a config that enables profiling
the config file `qwen2-5-7b-p-fsdp1-tp2-r-tp2-pp1-grpo.toml` has already been set to allow for profiling and will profile 1 training step on rank 0 of a policy replica.
```
python tools/launch_all.py --config ./configs/qwen2-5/qwen2-5-7b-p-fsdp1-tp2-r-tp2-pp1-grpo.toml --debug
```
2. Specify which policy replica to run the profiling

```
python profile_cmd.py -c localhost:8000 -r d86e828c-5b84-42a1-8de4-b23e247a7352
```

3. Get the trace file.

From the local disk, we could find the trace file: `outputs/qwen2-5-7b-p-fsdp1-tp2-r-tp2-pp1-grpo/profile_trace/d86e828c-5b84-42a1-8de4-b23e247a7352_0/trace.json`.


### For SFT

SFT profiling is also supported. Unlike RL, the profiler will start automatically at the beginning of the training. And then it will profile like normal.

When active steps are reached, the profiler will stop and save the trace file.

For example:

```
python tools/launch_all.py --config ./configs/qwen2-5/qwen2-5-7b-pp2-tp2-sft.toml --debug
```

Then after the active steps, the trace file will be saved to `outputs/qwen2-5-7b-sft-tp2-pp2-sft/profile_trace/b505e343-58ca-4459-af8c-0dbf9b1188b2_0/trace.json`.
