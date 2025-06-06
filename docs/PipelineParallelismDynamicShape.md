# Pipeline Parallelism Dynamic Shape: Implementation Overview

## Introduction

Pipeline parallelism is a distributed training technique that splits a model into multiple stages, each running on a different device or process. Traditionally, pipeline parallelism assumes static input shapes for efficient buffer allocation and communication. However, supporting **dynamic shapes** (e.g., variable sequence lengths) is crucial for modern NLP workloads and efficient memory usage.

The following information is sourced from the [PyTorch documentation](https://docs.pytorch.org/docs/stable/distributed.pipelining.html):
> A PipelineStage needs to know the input and output shapes for the stage model, so that it can correctly allocate communication buffers. The shapes must be static, e.g. at runtime the shapes can not change from step to step. A class PipeliningShapeError will be raised if runtime shapes do not match the expected shapes. When composing with other paralleisms or applying mixed precision, these techniques must be taken into account so the PipelineStage knows the correct shape (and dtype) for the output of the stage module at runtime.

**Echo** extends torch pipeline parallelism to support dynamic shapes, allowing the pipeline to adapt to varying input sequence lengths at runtime. The implementation is modular and integrates with both training and validation workflows.

---

## Key Components

### 1. **Dynamic Shape Metadata Tracking**

- **Attributes** such as `input_seq_dim`, `output_seq_dim`, `position_ids_seq_dim`, `init_position_ids_shape`, and `seq_len_multiple` are introduced in the pipeline schedule classes (`Schedule1F1B`, `ScheduleGPipe`).
- These attributes track which dimension of the input/output tensors corresponds to the sequence length and what the initial shape is, enabling the system to detect and adapt to shape changes.

### 2. **Shape Inference and Update Logic**

- The core logic for dynamic shape support is implemented in the functions:
  - `infer_seq_dim(self, position_ids_shape)`
  - `update_stage(self, position_ids_shape)`
- **`infer_seq_dim`**:
  - Compares the current `position_ids` shape with the initial shape to find the dimension where the sequence length changes.
  - Updates `input_seq_dim` and `output_seq_dim` accordingly.
- **`update_stage`**:
  - When a new batch with a different sequence length arrives, this function updates the input/output meta tensors to match the new shape.
  - It reconstructs the pipeline's forward and backward infrastructure to accommodate the new shapes.

### 3. **Stage Re-initialization**

- If the shape inference fails (e.g., more than one dimension changes, or the sequence dimension cannot be inferred), the pipeline stage is cleared and re-initialized to avoid inconsistent states.
- This is controlled by the `clear_stage_enabled` flag and the `_clear_stage()` method.

### 4. **Integration with Training and Validation**

- The `step` method in both `Schedule1F1B` and `ScheduleGPipe` is responsible for:
  - Splitting the input batch into microbatches.
  - Detecting and handling dynamic shape changes.
  - Running the microbatches through the pipeline.
- The `SFTTrainer` class in `sft_trainer.py` uses these schedules for both training and validation, passing the necessary dynamic shape information (e.g., `position_ids`, `seq_len_multiple`) to the pipeline scheduler.

### 5. **Microbatch Handling**

- Inputs, targets, and other relevant tensors are split into microbatches.
- For each microbatch, the pipeline checks if the shape has changed and updates the meta information if necessary before proceeding with forward and backward passes.

---

## Workflow Summary

1. **Initialization**:
   - The initial shape of `position_ids` is recorded.
   - The sequence length multiple (for padding/alignment) is set.

2. **Per-Batch Processing**:
   - For each new batch, the current `position_ids` shape is compared to the initial shape.
   - If the sequence length dimension has changed, the pipeline updates its input/output meta tensors and reconstructs the necessary communication buffers.

3. **Microbatch Execution**:
   - The batch is split into microbatches.
   - Each microbatch is processed according to the updated meta information, ensuring correct communication and computation across pipeline stages.

4. **Stage Clearing (if needed)**:
   - If the system cannot reliably infer the sequence dimension or encounters unsupported scenarios (e.g., multiple input/output metas), it clears and re-initializes the pipeline stage to maintain consistency.

---

## Design Highlights

- **Automatic Sequence Dimension Detection**:
  The system automatically infers which dimension of the input/output tensors corresponds to the sequence length, minimizing manual configuration.

- **Meta Tensor Management**:
  Input and output meta tensors are dynamically updated to match the current batch's shape, ensuring that buffer allocation and communication are always correct.

- **Seamless Training/Validation Integration**:
  The same dynamic shape logic is used for both training and validation, allowing for flexible and efficient evaluation without code duplication.

- **Robustness**:
  The implementation includes checks and fallbacks (e.g., stage clearing) to handle edge cases and maintain pipeline integrity.

---

## Note
Pipeline Parallelism Dynamic Shape is enabled by default. You can disable it by adding one line in *.toml config file:
```
[policy.parallelism]
...
pp_dynamic_shape = false
...
```