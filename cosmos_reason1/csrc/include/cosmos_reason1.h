// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <nccl.h>
#include <torch/torch.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#define NCCL_CHECK(cmd)                                                       \
  do {                                                                        \
    ncclResult_t r = cmd;                                                     \
    if (r != ncclSuccess && r != ncclInProgress) {                            \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,           \
             ncclGetErrorString(r));                                          \
      throw std::runtime_error(std::string("[COSMOS] NCCL error: ") +         \
                               ncclGetErrorString(r) + " " + __FILE__ + ":" + \
                               std::to_string(__LINE__) + " \n");             \
    }                                                                         \
  } while (0)

template <typename T>
void check(T result, char const* const func, const char* const file,
           int const line) {
  if (result) {
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") +
                             (cudaGetErrorString(result)) + " " + file + ":" +
                             std::to_string(line) + " \n");
  }
}

#define COSMOS_CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)

namespace cosmos_reason1 {

inline ncclDataType_t torch_dtype_to_nccl_dtype(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat:
      return ncclFloat;
    case torch::kHalf:
      return ncclHalf;
    case torch::kBFloat16:
      return ncclBfloat16;
    case torch::kDouble:
      return ncclDouble;
    case torch::kInt:
      return ncclInt;
    case torch::kLong:
      return ncclInt64;
    case torch::kUInt8:
      return ncclChar;
    case torch::kInt8:
      return ncclChar;
    default:
      TORCH_CHECK(false, "Unsupported tensor dtype for NCCL send");
  }
}

cudaStream_t getCurrentCUDAStream();

void nccl_abort(int64_t comm_idx);

[[noreturn]] inline void throwRuntimeError(const char* const file,
                                           int const line,
                                           std::string const& info = "") {
  throw std::runtime_error(std::string("[COSMOS][ERROR] ") + info +
                           " Assertion fail: " + file + ":" +
                           std::to_string(line) + " \n");
}

inline void myAssert(bool result, const char* const file, int const line,
                     std::string const& info = "") {
  if (!result) {
    throwRuntimeError(file, line, info);
  }
}

#define COSMOS_CHECK(val) myAssert(val, __FILE__, __LINE__)
#define COSMOS_CHECK_WITH_INFO(val, info)                                 \
  do {                                                                    \
    bool is_valid_val = (val);                                            \
    if (!is_valid_val) {                                                  \
      cosmos_reason1::myAssert(is_valid_val, __FILE__, __LINE__, (info)); \
    }                                                                     \
  } while (0)
}  // namespace cosmos_reason1
