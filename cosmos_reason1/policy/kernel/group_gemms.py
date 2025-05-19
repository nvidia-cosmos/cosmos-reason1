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

import torch
import torch.nn as nn

try:
    import deep_gemm

    DEEPGEMM_AVAILABLE = True
except ImportError:
    DEEPGEMM_AVAILABLE = False

if DEEPGEMM_AVAILABLE:
    import dsgemm_kernels
    import dsgemm_utils


# Strategy base class for GroupGEMM implementations
class GroupGEMMStrategy:
    """Base class for group gemm strategies"""

    def __init__(self, custom_activation):
        self.activation_function = custom_activation

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prepare expert weights, including prescaling

        Args:
            all_weights: List of weight tensors from each expert
            submod_name: Name of the submodule (e.g., 'gate_proj', 'up_proj', 'down_proj')
            module: The parent module that will store the arranged weights

        Returns:
            Tensor: The arranged weights in the format required by the specific strategy
        """

        raise NotImplementedError("Requires arrange_expert_weights method")

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute the group gemm operation

        Args:
            contig_tokens: The input tokens, arranged contiguously by expert
            m_sizes: Sizes of each group
            m_offsets: Offsets of each group
            module: The MoE module containing weights and parameters

        Returns:
            The processed tokens
        """
        raise NotImplementedError("GroupGEMM strategy must implement execute method")

    @staticmethod
    def is_available() -> bool:
        """Check if this strategy is available on the current system"""
        return False


# ========= Implementations ===================

__all__ = [
    "DSGroupGEMM",
    "TorchBF16GroupGEMM",
]


class TorchBF16GroupGEMM(GroupGEMMStrategy):
    """Implementation for PyTorch native BF16  _grouped_mm"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage"""
        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get weights
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Run first two GEMMs (gate and up projections)
        gate_proj = torch._grouped_mm(
            contig_tokens,
            w_gate.transpose(-2, -1),
            m_offsets,
            out_dtype=torch.bfloat16,
        )
        up_proj = torch._grouped_mm(
            contig_tokens,
            w_up.transpose(-2, -1),
            m_offsets,
            out_dtype=torch.bfloat16,
        )

        # Apply activation
        hidden_outputs = self.activation_function(gate_proj) * up_proj

        # Run the third GEMM (down projection)
        hidden_outputs = torch._grouped_mm(
            hidden_outputs,
            w_down.transpose(-2, -1),
            m_offsets,
            out_dtype=torch.bfloat16,
        )

        return hidden_outputs


class DSGroupGEMM(GroupGEMMStrategy):
    """Implementation using DeepGEMM with FP8 quantization"""

    def __init__(self, custom_activation, use_triton_quant=True):
        self.activation_function = custom_activation
        self.use_triton_quant = use_triton_quant

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage"""
        combined_weights = torch.stack(all_weights)

        fp8, scales = dsgemm_utils.prepare_fp8_weight(combined_weights)

        # prescale weights
        # TODO - this creates 2 sets of weights, need to resolve this for traiing aspect.
        module.register_parameter(
            f"{submod_name}_fp8",
            nn.Parameter(
                fp8,
            ),
        )

        module.register_parameter(
            f"{submod_name}_scales",
            nn.Parameter(
                scales,
            ),
        )

        return combined_weights

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get only valid tokens
        valid_tokens = contig_tokens[: m_offsets[-1]]

        # Create indices from offsets without CPU-GPU sync
        m_indices = dsgemm_utils.create_indices_from_offsets_nosync(m_offsets)

        # Get expert weights for all projections
        gate_proj_weight_fp8 = module.get_parameter("gate_proj_fp8")
        gate_proj_scales = module.get_parameter("gate_proj_scales")
        up_proj_weight_fp8 = module.get_parameter("up_proj_fp8")
        up_proj_scales = module.get_parameter("up_proj_scales")
        down_proj_weight_fp8 = module.get_parameter("down_proj_fp8")
        down_proj_scales = module.get_parameter("down_proj_scales")

        # Get dimensions
        m_actual_tokens = valid_tokens.shape[0]
        intermediate_size = module.get_parameter("gate_proj_weight").shape[1]
        hidden_size = module.get_parameter("down_proj_weight").shape[1]

        # Allocate output buffers
        gate_proj_out = torch.empty(
            (m_actual_tokens, intermediate_size),
            device=contig_tokens.device,
            dtype=contig_tokens.dtype,
        )
        up_proj_out = torch.empty_like(gate_proj_out)

        # Allocate output buffer for down projection
        down_proj_out = torch.empty(
            (m_actual_tokens, hidden_size),
            device=contig_tokens.device,
            dtype=contig_tokens.dtype,
        )

        # Prepare input in FP8 format (shared by gate and up projections)
        if self.use_triton_quant:
            gate_up_input = dsgemm_kernels.groupwise_activation_quant(valid_tokens)
        else:
            gate_up_input = dsgemm_utils.prepare_fp8_input(valid_tokens)

        # Run first GEMM (gate projection)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            gate_up_input,
            (gate_proj_weight_fp8, gate_proj_scales),
            gate_proj_out,
            m_indices,
        )

        # Run second GEMM (up projection)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            gate_up_input,
            (up_proj_weight_fp8, up_proj_scales),
            up_proj_out,
            m_indices,
        )

        # Apply activation
        hidden_states = self.activation_function(gate_proj_out) * up_proj_out

        # Run third GEMM (down projection)
        if self.use_triton_quant:
            hidden_states_quantized = dsgemm_kernels.groupwise_activation_quant(
                hidden_states
            )
        else:
            hidden_states_quantized = dsgemm_utils.prepare_fp8_input(hidden_states)

        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            hidden_states_quantized,
            (down_proj_weight_fp8, down_proj_scales),
            down_proj_out,
            m_indices,
        )

        # Copy results back to contig_tokens
        contig_tokens[: m_offsets[-1]] = down_proj_out
        return contig_tokens

    @staticmethod
    def is_available() -> bool:
        return DEEPGEMM_AVAILABLE
