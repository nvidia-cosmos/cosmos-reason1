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

#include "cosmos_reason1.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

#include "util.h"
#include "watchdog_tls.h"

namespace py = pybind11;
static std::vector<ncclComm_t> shared_comms;

namespace cosmos_reason1 {

namespace nccl {
static const int64_t DEFAULT_TIMEOUT_MS = ([]() -> int {
  if (std::getenv("COSMOS_NCCL_TIMEOUT_MS") != nullptr) {
    try {
      std::string groupsize =
          std::string(std::getenv("COSMOS_NCCL_TIMEOUT_MS"));
      return std::stoi(groupsize);
    } catch (...) {
      std::cout
          << "Invalid COSMOS_NCCL_TIMEOUT_MS, using default value `600000`"
          << std::endl;
    }
  }
  return 600000;
})();
;

struct AsyncNCCLOP {
  int64_t timeout_ms;
  cudaStream_t stream;
  std::function<ncclComm_t()> functor;
  std::atomic<bool> out_timeout;

  std::atomic<bool> consumed_;

  AsyncNCCLOP(int64_t timeout_ms, cudaStream_t stream,
              std::function<ncclComm_t()> functor, bool out_timeout_)
      : timeout_ms(timeout_ms), stream(stream), functor(functor) {
    out_timeout.store(out_timeout_);
    consumed_.store(false);
  }
};

void async_enqueue_timeout(ncclComm_t comm, int64_t timeout_ms) {
  auto start_time = std::chrono::steady_clock::now();
  auto timeout = start_time + std::chrono::milliseconds(timeout_ms);
  ncclResult_t result;

  do {
    ncclCommGetAsyncError(comm, &result);
    if (result == ncclSuccess) {
      return;
    } else if (result == ncclInProgress) {
      // non-blocking enqueue is in progress
      // Have to be blocked before next cuda kernel is launched
      if (std::chrono::steady_clock::now() > timeout) {
        break;
      }
    } else {
      // other error
      break;
    }

  } while (result == ncclInProgress);

  ncclCommAbort(comm);
  throw std::runtime_error("Time out");
}

static std::queue<std::shared_ptr<AsyncNCCLOP>> async_nccl_ops;
static std::mutex async_nccl_ops_mutex;

void async_enqueue(int64_t timeout_ms, std::function<ncclComm_t()> functor) {
  {
    // Forbid cudagraph capture mode because timeout check will not work
    cudaStreamCaptureStatus status;
    cudaStreamIsCapturing(getCurrentCUDAStream(), &status);
    if (status != cudaStreamCaptureStatusNone) {
      throw std::runtime_error("Cudagraph capture mode is not allowed");
    }
  }

  std::unique_lock<std::mutex> lock(async_nccl_ops_mutex);
  auto op = std::make_shared<AsyncNCCLOP>(timeout_ms, getCurrentCUDAStream(),
                                          functor, false);
  defer {
    if (lock.owns_lock()) {
      lock.unlock();
    }
  };

  async_nccl_ops.push(op);
  lock.unlock();

  while (!op->consumed_.load()) {
    auto out_timeout = op->out_timeout.load();
    if (out_timeout) {
      throw std::runtime_error("NCCL: non-blocking enqueue timed out");
    }
  }
}

void worker(int current_device_index) {
  try {
    cudaSetDevice(current_device_index);
    while (true) {
      std::unique_lock<std::mutex> lock(async_nccl_ops_mutex);
      defer {
        if (lock.owns_lock()) {
          lock.unlock();
        }
      };
      if (!async_nccl_ops.empty()) {
        auto op = async_nccl_ops.front();
        async_nccl_ops.pop();
        lock.unlock();
        try {
          ncclComm_t comm = op->functor();
          // Ensure the kernel is dispatched to the stream
          async_enqueue_timeout(comm, op->timeout_ms);
        } catch (const std::runtime_error& e) {
          op->out_timeout.store(true);
        }
        op->consumed_.store(true);
      } else {
        // Spin until the operation is consumed
      }
    }
  } catch (const std::exception& e) {
    std::cout << "NCCL: error on worker : " << e.what() << std::endl;
  }
}

}  // namespace nccl

cudaStream_t getCurrentCUDAStream() {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  return stream;
}

torch::Tensor from_blob(int64_t data_ptr, std::vector<int64_t> shape,
                        torch::Tensor sample) {
  auto options = torch::TensorOptions();
  options = options.dtype(sample.dtype()).device(sample.device());
  void* ptr = reinterpret_cast<void*>(data_ptr);
  return torch::from_blob(ptr, shape, options);
}

std::vector<int64_t> create_nccl_uid() {
  ncclUniqueId uid;
  NCCL_CHECK(ncclGetUniqueId(&uid));
  std::vector<int64_t> result(128);
  for (int i = 0; i < 128; i++) {
    result[i] = uid.internal[i];
  }
  return result;
}

int64_t create_nccl_comm(std::vector<int64_t> uid_chars, int64_t rank,
                         int64_t world_size) {
  static std::once_flag once_flag;
  std::call_once(once_flag, []() {
    // Get current device
    int current_device_index;
    cudaGetDevice(&current_device_index);
    std::thread nccl_worker(nccl::worker, current_device_index);
    nccl_worker.detach();
  });
  ncclUniqueId uid;
  for (int i = 0; i < 128; i++) {
    uid.internal[i] = uid_chars[i];
  }
  static_assert(sizeof(ncclUniqueId) == 128,
                "ncclUniqueId size is not 128 bytes");
  ncclComm_t comm;

  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#fault-tolerance
  // To abort the nccl communicator safely, we need to set the blocking to 0

  auto functor = [&]() -> ncclComm_t {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    NCCL_CHECK(ncclCommInitRankConfig(&comm, world_size, uid, rank, &config));
    return comm;
  };

  nccl::async_enqueue(nccl::DEFAULT_TIMEOUT_MS, functor);
  shared_comms.push_back(comm);
  return shared_comms.size() - 1;
}

ncclComm_t get_nccl_comm(int64_t idx) {
  COSMOS_CHECK_WITH_INFO(idx >= 0 && idx < shared_comms.size(),
                         "Invalid NCCL communicator index");
  return shared_comms[idx];
}

int nccl_get_comm_count(int64_t comm_idx) {
  auto comm = get_nccl_comm(comm_idx);
  int nranks;
  NCCL_CHECK(ncclCommCount(comm, &nranks));
  // This is usually not needed, but we do it to make sure state is consistent
  return nranks;
}

void nccl_broadcast(torch::Tensor tensor, int64_t rank, int64_t comm_idx) {
  TORCH_CHECK(tensor.is_cuda(), "Tensor must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
  TORCH_CHECK(tensor.numel() > 0,
              "Tensor must have non-zero number of elements");

  int tensor_device_index = tensor.device().index();
  int current_device_index;
  COSMOS_CUDA_CHECK(cudaGetDevice(&current_device_index));
  TORCH_CHECK(current_device_index == tensor_device_index,
              "Current CUDA device does not match tensor's device. Make sure "
              "to set the right device before using NCCL.");

  ncclComm_t comm = get_nccl_comm(comm_idx);
  size_t count = tensor.numel();
  ncclDataType_t dtype = torch_dtype_to_nccl_dtype(tensor.scalar_type());

  void* data_ptr = tensor.data_ptr();

  TORCH_CHECK(data_ptr != nullptr,
              "Tensor data_ptr is null (tensor likely not materialized)");

  cudaStream_t stream = getCurrentCUDAStream();

  auto functor = [&]() -> ncclComm_t {
    NCCL_CHECK(ncclBroadcast(data_ptr,  // sendbuff
                             data_ptr,  // recvbuff
                             count, dtype, rank, comm, stream));
    return comm;
  };

  WatchdogTLS::add_action(WatchdogAction{
      .name_ = "nccl_broadcast",
      .comm_idx_ = comm_idx,
      .abort_func_ = nccl_abort,
  });

  nccl::async_enqueue(nccl::DEFAULT_TIMEOUT_MS, functor);
}

void nccl_send(torch::Tensor tensor, int64_t peer, int64_t comm_idx) {
  TORCH_CHECK(tensor.is_cuda(), "Tensor must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
  TORCH_CHECK(tensor.numel() > 0,
              "Tensor must have non-zero number of elements");

  int tensor_device_index = tensor.device().index();
  int current_device_index;
  COSMOS_CUDA_CHECK(cudaGetDevice(&current_device_index));
  TORCH_CHECK(current_device_index == tensor_device_index,
              "Current CUDA device does not match tensor's device.");

  ncclComm_t comm = get_nccl_comm(comm_idx);
  size_t count = tensor.numel();
  ncclDataType_t dtype = torch_dtype_to_nccl_dtype(tensor.scalar_type());

  void* data_ptr = tensor.data_ptr();
  TORCH_CHECK(data_ptr != nullptr, "Tensor data_ptr is null");

  cudaStream_t stream = getCurrentCUDAStream();

  auto functor = [&]() {
    NCCL_CHECK(ncclSend(data_ptr, count, dtype, peer, comm, stream));
    return comm;
  };

  WatchdogTLS::add_action(WatchdogAction{
      .name_ = "nccl_send",
      .comm_idx_ = comm_idx,
      .abort_func_ = nccl_abort,
  });

  nccl::async_enqueue(nccl::DEFAULT_TIMEOUT_MS, functor);
}

void nccl_recv(torch::Tensor tensor, int64_t peer, int64_t comm_idx) {
  TORCH_CHECK(tensor.is_cuda(), "Tensor must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
  TORCH_CHECK(tensor.numel() > 0,
              "Tensor must have non-zero number of elements");

  int tensor_device_index = tensor.device().index();
  int current_device_index;
  COSMOS_CUDA_CHECK(cudaGetDevice(&current_device_index));
  TORCH_CHECK(current_device_index == tensor_device_index,
              "Current CUDA device does not match tensor's device.");

  ncclComm_t comm = get_nccl_comm(comm_idx);
  size_t count = tensor.numel();
  ncclDataType_t dtype = torch_dtype_to_nccl_dtype(tensor.scalar_type());

  void* data_ptr = tensor.data_ptr();
  TORCH_CHECK(data_ptr != nullptr, "Tensor data_ptr is null");

  cudaStream_t stream = getCurrentCUDAStream();

  auto functor = [&]() -> ncclComm_t {
    NCCL_CHECK(ncclRecv(data_ptr, count, dtype, peer, comm, stream));
    return comm;
  };

  WatchdogTLS::add_action(WatchdogAction{
      .name_ = "nccl_recv",
      .comm_idx_ = comm_idx,
      .abort_func_ = nccl_abort,
  });
  nccl::async_enqueue(nccl::DEFAULT_TIMEOUT_MS, functor);
}

void nccl_allreduce(torch::Tensor sendbuff, torch::Tensor recvbuff, int64_t op,
                    int64_t comm_idx) {
  TORCH_CHECK(sendbuff.is_cuda(), "Send Tensor must be a CUDA tensor");
  TORCH_CHECK(sendbuff.is_contiguous(), "Send Tensor must be contiguous");
  TORCH_CHECK(sendbuff.numel() > 0,
              "Send Tensor must have non-zero number of elements");
  TORCH_CHECK(recvbuff.is_cuda(), "Recv Tensor must be a CUDA tensor");
  TORCH_CHECK(recvbuff.is_contiguous(), "Recv Tensor must be contiguous");
  TORCH_CHECK(recvbuff.numel() > 0,
              "Recv Tensor must have non-zero number of elements");
  TORCH_CHECK(sendbuff.device() == recvbuff.device(),
              "sendbuff and recvbuff must be on the same device");
  TORCH_CHECK(sendbuff.numel() == recvbuff.numel(),
              "sendbuff and recvbuff must have the same number of elements");
  TORCH_CHECK(sendbuff.scalar_type() == recvbuff.scalar_type(),
              "sendbuff and recvbuff must have the same dtype");

  int tensor_device_index = sendbuff.device().index();
  int current_device_index;
  COSMOS_CUDA_CHECK(cudaGetDevice(&current_device_index));
  TORCH_CHECK(current_device_index == tensor_device_index,
              "Current CUDA device does not match tensor's device.");

  ncclComm_t comm = get_nccl_comm(comm_idx);
  auto count = sendbuff.numel();
  ncclDataType_t dtype = torch_dtype_to_nccl_dtype(sendbuff.scalar_type());

  ncclRedOp_t red_op;
  switch (op) {
    case 0:
      red_op = ncclSum;
      break;
    case 1:
      red_op = ncclProd;
      break;
    case 2:
      red_op = ncclMax;
      break;
    case 3:
      red_op = ncclMin;
      break;
    case 4:
      red_op = ncclAvg;
      break;
    default:
      TORCH_CHECK(false, "Unsupported reduction operation for NCCL allreduce");
  }

  void* send_ptr = sendbuff.data_ptr();
  TORCH_CHECK(send_ptr != nullptr, "Send tensor data_ptr is null");

  void* recv_ptr = recvbuff.data_ptr();
  TORCH_CHECK(recv_ptr != nullptr, "Recv tensor data_ptr is null");

  cudaStream_t stream = getCurrentCUDAStream();

  auto functor = [&]() -> ncclComm_t {
    ncclAllReduce(send_ptr, recv_ptr, count, dtype, red_op, comm, stream);
    return comm;
  };

  WatchdogTLS::add_action(WatchdogAction{
      .name_ = "nccl_allreduce",
      .comm_idx_ = comm_idx,
      .abort_func_ = nccl_abort,
  });
  nccl::async_enqueue(nccl::DEFAULT_TIMEOUT_MS, functor);
}

void nccl_alltoall(torch::Tensor sendbuff, torch::Tensor recvbuff,
                   int64_t comm_idx) {
  TORCH_CHECK(sendbuff.is_cuda(), "Send Tensor must be a CUDA tensor");
  TORCH_CHECK(sendbuff.is_contiguous(), "Send Tensor must be contiguous");
  TORCH_CHECK(sendbuff.numel() > 0,
              "Send Tensor must have non-zero number of elements");
  TORCH_CHECK(recvbuff.is_cuda(), "Recv Tensor must be a CUDA tensor");
  TORCH_CHECK(recvbuff.is_contiguous(), "Recv Tensor must be contiguous");
  TORCH_CHECK(recvbuff.numel() > 0,
              "Recv Tensor must have non-zero number of elements");
  TORCH_CHECK(sendbuff.device() == recvbuff.device(),
              "sendbuff and recvbuff must be on the same device");
  TORCH_CHECK(sendbuff.numel() == recvbuff.numel(),
              "sendbuff and recvbuff must have the same number of elements");
  TORCH_CHECK(sendbuff.scalar_type() == recvbuff.scalar_type(),
              "sendbuff and recvbuff must have the same dtype");

  int tensor_device_index = sendbuff.device().index();
  int current_device_index;
  COSMOS_CUDA_CHECK(cudaGetDevice(&current_device_index));
  TORCH_CHECK(current_device_index == tensor_device_index,
              "Current CUDA device does not match tensor's device.");

  ncclComm_t comm = get_nccl_comm(comm_idx);
  auto total_size = sendbuff.numel();
  ncclDataType_t dtype = torch_dtype_to_nccl_dtype(sendbuff.scalar_type());

  int world_size;
  NCCL_CHECK(ncclCommCount(comm, &world_size));

  int rank;
  NCCL_CHECK(ncclCommUserRank(comm, &rank));

  size_t count = total_size / world_size;

  void* send_ptr = sendbuff.data_ptr();
  TORCH_CHECK(send_ptr != nullptr, "Send tensor data_ptr is null");

  void* recv_ptr = recvbuff.data_ptr();
  TORCH_CHECK(recv_ptr != nullptr, "Recv tensor data_ptr is null");

  cudaStream_t stream = getCurrentCUDAStream();

  auto functor = [&]() -> ncclComm_t {
    size_t rankOffset = count * sendbuff.element_size();
    NCCL_CHECK(ncclGroupStart());
    for (int r = 0; r < world_size; r++) {
      NCCL_CHECK(ncclSend(((char*)send_ptr) + r * rankOffset, count, dtype, r,
                          comm, stream));
      NCCL_CHECK(ncclRecv(((char*)recv_ptr) + r * rankOffset, count, dtype, r,
                          comm, stream));
    }
    NCCL_CHECK(ncclGroupEnd());
    return comm;
  };

  WatchdogTLS::add_action(WatchdogAction{
      .name_ = "nccl_alltoall",
      .comm_idx_ = comm_idx,
      .abort_func_ = nccl_abort,
  });
  nccl::async_enqueue(nccl::DEFAULT_TIMEOUT_MS, functor);
}

void nccl_abort(int64_t comm_idx) {
  auto comm = get_nccl_comm(comm_idx);
  if (comm == nullptr) {
    return;
  }
  NCCL_CHECK(ncclCommAbort(comm));
}

int64_t nccl_timeout_in_ms() { return nccl::DEFAULT_TIMEOUT_MS; }

void watchdog_enter() { WatchdogTLS::new_context(); }

void watchdog_exit(bool abort) { WatchdogTLS::pop_context(abort); }

}  // namespace cosmos_reason1

PYBIND11_MODULE(_cpp, m) {
  m.doc() = "Cosmos C++/CUDA extension";
  m.def("create_nccl_comm", &cosmos_reason1::create_nccl_comm,
        py::call_guard<py::gil_scoped_release>(), "Create a NCCL communicator");
  // m.def("get_nccl_comm", &cosmos_reason1::get_nccl_comm, "Get a NCCL
  // communicator");
  m.def("from_blob", &cosmos_reason1::from_blob, "Create a tensor from a blob");
  m.def("create_nccl_uid", &cosmos_reason1::create_nccl_uid,
        "Create a NCCL unique ID");
  m.def("get_nccl_comm_count", &cosmos_reason1::nccl_get_comm_count,
        py::arg("comm_idx"),
        "Get the number of ranks in the NCCL communicator");
  m.def("nccl_broadcast", &cosmos_reason1::nccl_broadcast, py::arg("tensor"),
        py::arg("rank"), py::arg("comm_idx"),
        py::call_guard<py::gil_scoped_release>(),
        R"pbdoc(
            Perform an NCCL broadcast on the given NCCL group.

            Args:
                tensor (torch.Tensor): Tensor to broadcast (must be on CUDA).
                root (int): Root rank in the communicator.
                comm_idx (int): Index of the communicator (created by `create_nccl_comm`).
        )pbdoc");

  m.def("nccl_send", &cosmos_reason1::nccl_send, py::arg("tensor"),
        py::arg("peer"), py::arg("comm_idx"),
        py::call_guard<py::gil_scoped_release>(),
        R"pbdoc(
            Perform an NCCL point-to-point send operation.

            Args:
                tensor (torch.Tensor): Tensor to send (must be CUDA and contiguous).
                peer (int): Rank to send to.
                comm_idx (int): Communicator index.
        )pbdoc");

  m.def("nccl_recv", &cosmos_reason1::nccl_recv, py::arg("tensor"),
        py::arg("peer"), py::arg("comm_idx"),
        py::call_guard<py::gil_scoped_release>(),
        R"pbdoc(
            Perform an NCCL point-to-point recv operation.

            Args:
                tensor (torch.Tensor): Tensor to receive into (must be CUDA and contiguous).
                peer (int): Rank to receive from.
                comm_idx (int): Communicator index.
        )pbdoc");

  m.def("nccl_allreduce", &cosmos_reason1::nccl_allreduce, py::arg("sendbuff"),
        py::arg("recvbuff"), py::arg("op"), py::arg("comm_idx"),
        py::call_guard<py::gil_scoped_release>(),
        R"pbdoc(
            Perform an NCCL allreduce operation.

            Args:
                sendbuff (torch.Tensor): Tensor to send (must be CUDA and contiguous).
                recvbuff (torch.Tensor): Tensor to receive into (must be CUDA and contiguous).
                op (int): Reduction operation (0: sum, 1: prod, 2: max, 3: min, 4: avg).
                comm_idx (int): Communicator index.
        )pbdoc");
  m.def("nccl_alltoall", &cosmos_reason1::nccl_alltoall, py::arg("sendbuff"),
        py::arg("recvbuff"), py::arg("comm_idx"),
        R"pbdoc(
          Perform an NCCL alltoall operation.

          Args:
              sendbuff (torch.Tensor): Tensor to send in alltoall (must be CUDA and contiguous).
              recvbuff (torch.Tensor): Tensor to receive into in altoall (must be CUDA and contiguous).
              comm_idx (int): Communicator index.
      )pbdoc");

  m.def("nccl_timeout_in_ms", &cosmos_reason1::nccl_timeout_in_ms,
        "Get the timeout for NCCL operations in milliseconds");
  m.def("nccl_abort", &cosmos_reason1::nccl_abort, py::arg("comm_idx"),
        R"pbdoc(
            Abort the NCCL communicator.
        )pbdoc");
  m.def("watchdog_enter", &cosmos_reason1::watchdog_enter,
        "Enter the watchdog context");
  m.def("watchdog_exit", &cosmos_reason1::watchdog_exit, py::arg("abort"),
        "Exit the watchdog context");
}
