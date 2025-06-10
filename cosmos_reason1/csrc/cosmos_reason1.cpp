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

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "util.h"
#include "watchdog_tls.h"

namespace py = pybind11;
static std::atomic<int64_t> comm_idx_counter = 0;
static std::unordered_map<int64_t, ncclComm_t> shared_comms;

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
          << "Invalid COSMOS_NCCL_TIMEOUT_MS, using default value `500000`"
          << std::endl;
    }
  }
  return 500000;
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

void async_enqueue(cudaStream_t stream, int64_t timeout_ms,
                   std::function<ncclComm_t()> functor) {
  {
    // Forbid cudagraph capture mode because timeout check will not work
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (stream != nullptr) {
      cudaStreamIsCapturing(stream, &status);
    }
    if (status != cudaStreamCaptureStatusNone) {
      throw std::runtime_error("Cudagraph capture mode is not allowed");
    }
  }

  std::unique_lock<std::mutex> lock(async_nccl_ops_mutex);
  auto op = std::make_shared<AsyncNCCLOP>(timeout_ms, stream, functor, false);
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

// ncclDtypeSize is reverse of ncclDatatypeToString @ nccl:/src/collectives.cc
size_t ncclDTypeSize(ncclDataType_t dtype) {
  switch (dtype) {
    case ncclInt8:
    case ncclUint8:
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
      return 1;
    case ncclFloat16:
    case ncclBfloat16:
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      throw std::runtime_error("Unsupported dtype");
  }
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
                         int64_t world_size, int64_t timeout_ms) {
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

  nccl::async_enqueue(nullptr, timeout_ms, functor);
  int64_t comm_idx = comm_idx_counter.fetch_add(1);
  shared_comms.insert({comm_idx, comm});
  return comm_idx;
}

ncclComm_t get_nccl_comm(int64_t idx) {
  COSMOS_CHECK_WITH_INFO(shared_comms.find(idx) != shared_comms.end(),
                         "Invalid NCCL communicator index");
  return shared_comms.at(idx);
}

int nccl_get_comm_count(int64_t comm_idx) {
  auto comm = get_nccl_comm(comm_idx);
  int nranks;
  NCCL_CHECK(ncclCommCount(comm, &nranks));
  // This is usually not needed, but we do it to make sure state is consistent
  return nranks;
}

int64_t get_default_timeout_ms() {
  // We won't directly use nccl::DEFAULT_TIMEOUT_MS, because we should let user
  // choose the timeout value, and we don't want to change the default value of
  // nccl::DEFAULT_TIMEOUT_MS
  return nccl::DEFAULT_TIMEOUT_MS;
}

void nccl_broadcast(void* tensor, size_t count, ncclDataType_t dtype,
                    int64_t rank, int64_t comm_idx, cudaStream_t stream,
                    int64_t timeout_ms) {
  COSMOS_CHECK_WITH_INFO(
      tensor != nullptr,
      "Tensor data_ptr is null (tensor likely not materialized)");

  ncclComm_t comm = get_nccl_comm(comm_idx);

  auto functor = [&]() -> ncclComm_t {
    NCCL_CHECK(ncclBroadcast(tensor,  // sendbuff
                             tensor,  // recvbuff
                             count, dtype, rank, comm, stream));
    return comm;
  };

  WatchdogTLS::add_action(WatchdogAction{
      .name_ = "nccl_broadcast",
      .comm_idx_ = comm_idx,
      .abort_func_ = nccl_abort,
  });

  nccl::async_enqueue(stream, timeout_ms, functor);
}

void nccl_send(void* tensor, size_t count, ncclDataType_t dtype, int64_t peer,
               int64_t comm_idx, cudaStream_t stream, int64_t timeout_ms) {
  COSMOS_CHECK_WITH_INFO(
      tensor != nullptr,
      "Tensor data_ptr is null (tensor likely not materialized)");

  ncclComm_t comm = get_nccl_comm(comm_idx);

  auto functor = [&]() {
    NCCL_CHECK(ncclSend(tensor, count, dtype, peer, comm, stream));
    return comm;
  };

  WatchdogTLS::add_action(WatchdogAction{
      .name_ = "nccl_send",
      .comm_idx_ = comm_idx,
      .abort_func_ = nccl_abort,
  });

  nccl::async_enqueue(stream, timeout_ms, functor);
}

void nccl_recv(void* tensor, size_t count, ncclDataType_t dtype, int64_t peer,
               int64_t comm_idx, cudaStream_t stream, int64_t timeout_ms) {
  COSMOS_CHECK_WITH_INFO(
      tensor != nullptr,
      "Tensor data_ptr is null (tensor likely not materialized)");

  ncclComm_t comm = get_nccl_comm(comm_idx);

  auto functor = [&]() -> ncclComm_t {
    NCCL_CHECK(ncclRecv(tensor, count, dtype, peer, comm, stream));
    return comm;
  };

  WatchdogTLS::add_action(WatchdogAction{
      .name_ = "nccl_recv",
      .comm_idx_ = comm_idx,
      .abort_func_ = nccl_abort,
  });
  nccl::async_enqueue(stream, timeout_ms, functor);
}

void nccl_allreduce(void* sendbuff, void* recvbuff, size_t count,
                    ncclDataType_t dtype, ncclRedOp_t op, int64_t comm_idx,
                    cudaStream_t stream, int64_t timeout_ms) {
  COSMOS_CHECK_WITH_INFO(sendbuff != nullptr,
                         "Send Tensor must be a CUDA tensor");
  COSMOS_CHECK_WITH_INFO(recvbuff != nullptr,
                         "Recv Tensor must be a CUDA tensor");
  COSMOS_CHECK_WITH_INFO(count > 0,
                         "Tensor must have non-zero number of elements");

  ncclComm_t comm = get_nccl_comm(comm_idx);

  auto functor = [&]() -> ncclComm_t {
    NCCL_CHECK(
        ncclAllReduce(sendbuff, recvbuff, count, dtype, op, comm, stream));
    return comm;
  };

  WatchdogTLS::add_action(WatchdogAction{
      .name_ = "nccl_allreduce",
      .comm_idx_ = comm_idx,
      .abort_func_ = nccl_abort,
  });
  nccl::async_enqueue(stream, timeout_ms, functor);
}

void nccl_alltoall(void* sendbuff, void* recvbuff, size_t total_size,
                   ncclDataType_t dtype, int64_t comm_idx, cudaStream_t stream,
                   int64_t timeout_ms) {
  COSMOS_CHECK_WITH_INFO(sendbuff != nullptr,
                         "Send Tensor must be a CUDA tensor");
  COSMOS_CHECK_WITH_INFO(recvbuff != nullptr,
                         "Recv Tensor must be a CUDA tensor");
  COSMOS_CHECK_WITH_INFO(total_size > 0,
                         "Tensor must have non-zero number of elements");

  ncclComm_t comm = get_nccl_comm(comm_idx);

  int world_size;
  NCCL_CHECK(ncclCommCount(comm, &world_size));
  int rank;
  NCCL_CHECK(ncclCommUserRank(comm, &rank));

  size_t count = total_size / world_size;
  size_t dtype_size = ncclDTypeSize(dtype);

  auto functor = [&]() -> ncclComm_t {
    size_t rankOffset = count * dtype_size;
    NCCL_CHECK(ncclGroupStart());
    for (int r = 0; r < world_size; r++) {
      NCCL_CHECK(ncclSend(((char*)sendbuff) + r * rankOffset, count, dtype, r,
                          comm, stream));
      NCCL_CHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, dtype, r,
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
  nccl::async_enqueue(stream, timeout_ms, functor);
}

void nccl_abort(int64_t comm_idx) {
  if (shared_comms.find(comm_idx) == shared_comms.end()) {
    return;
  }

  auto comm = get_nccl_comm(comm_idx);
  if (comm == nullptr) {
    return;
  }
  NCCL_CHECK(ncclCommAbort(comm));
  shared_comms.erase(comm_idx);
}

void watchdog_enter() { WatchdogTLS::new_context(); }

void watchdog_exit(bool abort) { WatchdogTLS::pop_context(abort); }

}  // namespace cosmos_reason1

PYBIND11_MODULE(_cpp, m) {
  m.doc() = "Cosmos C++/CUDA extension";

  m.def("watchdog_enter", &cosmos_reason1::watchdog_enter,
        "Enter the watchdog context");
  m.def("watchdog_exit", &cosmos_reason1::watchdog_exit, py::arg("abort"),
        "Exit the watchdog context");

  m.def("create_nccl_comm", &cosmos_reason1::create_nccl_comm,
        py::arg("uid_chars"), py::arg("rank"), py::arg("world_size"),
        py::arg("timeout_ms") = cosmos_reason1::get_default_timeout_ms(),
        py::call_guard<py::gil_scoped_release>(), "Create a NCCL communicator");
  m.def("create_nccl_uid", &cosmos_reason1::create_nccl_uid,
        "Create a NCCL unique ID");
  m.def("nccl_abort", &cosmos_reason1::nccl_abort, py::arg("comm_idx"),
        R"pbdoc(
            Abort the NCCL communicator.
        )pbdoc");
  m.def("get_nccl_comm_count", &cosmos_reason1::nccl_get_comm_count,
        py::arg("comm_idx"),
        "Get the number of ranks in the NCCL communicator");
  m.def("get_default_timeout_ms", &cosmos_reason1::get_default_timeout_ms,
        "Get the default timeout value for NCCL operations");
  m.def(
      "nccl_broadcast",
      [](intptr_t tensor, size_t count, int dtype, int64_t rank,
         int64_t comm_idx, intptr_t stream, int64_t timeout_ms) {
        void* tensor_ptr = (void*)tensor;
        cudaStream_t stream_ptr = (cudaStream_t)stream;
        ncclDataType_t dtype_ = (ncclDataType_t)dtype;
        cosmos_reason1::nccl_broadcast(tensor_ptr, count, dtype_, rank,
                                       comm_idx, stream_ptr, timeout_ms);
      },
      py::arg("tensor"), py::arg("count"), py::arg("dtype"), py::arg("rank"),
      py::arg("comm_idx"), py::arg("stream"),
      py::arg("timeout_ms") = cosmos_reason1::get_default_timeout_ms(),
      py::call_guard<py::gil_scoped_release>(),
      R"pbdoc(
            Perform an NCCL broadcast on the given NCCL group.

            Args:
                tensor (void*): Tensor to broadcast (must be on CUDA).
                count (int): Number of elements to broadcast.
                dtype (ncclDataType_t): Data type of the tensor.
                rank (int): Root rank in the communicator.
                comm_idx (int): Index of the communicator (created by `create_nccl_comm`).
                stream (cudaStream_t): CUDA stream to use for the operation.
                timeout_ms (int): Timeout value for the operation.
        )pbdoc");

  m.def(
      "nccl_send",
      [](intptr_t tensor, size_t count, int dtype, int64_t peer,
         int64_t comm_idx, intptr_t stream, int64_t timeout_ms) {
        void* tensor_ptr = (void*)tensor;
        ncclDataType_t dtype_ = (ncclDataType_t)dtype;
        cudaStream_t stream_ptr = (cudaStream_t)stream;
        cosmos_reason1::nccl_send(tensor_ptr, count, dtype_, peer, comm_idx,
                                  stream_ptr, timeout_ms);
      },
      py::arg("tensor"), py::arg("count"), py::arg("dtype"), py::arg("peer"),
      py::arg("comm_idx"), py::arg("stream"),
      py::arg("timeout_ms") = cosmos_reason1::get_default_timeout_ms(),
      py::call_guard<py::gil_scoped_release>(),
      R"pbdoc(
            Perform an NCCL point-to-point send operation.

            Args:
                tensor (void*): Tensor to send (must be CUDA and contiguous).
                count (int): Number of elements to send.
                dtype (ncclDataType_t): Data type of the tensor.
                peer (int): Rank to send to.
                comm_idx (int): Communicator index.
                stream (cudaStream_t): CUDA stream to use for the operation.
                timeout_ms (int): Timeout value for the operation.
        )pbdoc");

  m.def(
      "nccl_recv",
      [](intptr_t tensor, size_t count, int dtype, int64_t peer,
         int64_t comm_idx, intptr_t stream, int64_t timeout_ms) {
        void* tensor_ptr = (void*)tensor;
        cudaStream_t stream_ptr = (cudaStream_t)stream;
        ncclDataType_t dtype_ = (ncclDataType_t)dtype;
        cosmos_reason1::nccl_recv(tensor_ptr, count, dtype_, peer, comm_idx,
                                  stream_ptr, timeout_ms);
      },
      py::arg("tensor"), py::arg("count"), py::arg("dtype"), py::arg("peer"),
      py::arg("comm_idx"), py::arg("stream"),
      py::arg("timeout_ms") = cosmos_reason1::get_default_timeout_ms(),
      py::call_guard<py::gil_scoped_release>(),
      R"pbdoc(
            Perform an NCCL point-to-point recv operation.

            Args:
                tensor (void*): Tensor to receive into (must be CUDA and contiguous).
                count (int): Number of elements to receive.
                dtype (ncclDataType_t): Data type of the tensor.
                peer (int): Rank to receive from.
                comm_idx (int): Communicator index.
                stream (cudaStream_t): CUDA stream to use for the operation.
                timeout_ms (int): Timeout value for the operation.
        )pbdoc");

  m.def(
      "nccl_allreduce",
      [](intptr_t sendbuff, intptr_t recvbuff, size_t count, int dtype, int op,
         int64_t comm_idx, intptr_t stream, int64_t timeout_ms) {
        void* sendbuff_ptr = (void*)sendbuff;
        void* recvbuff_ptr = (void*)recvbuff;
        ncclDataType_t dtype_ = (ncclDataType_t)dtype;
        ncclRedOp_t op_ = (ncclRedOp_t)op;
        cudaStream_t stream_ptr = (cudaStream_t)stream;
        cosmos_reason1::nccl_allreduce(sendbuff_ptr, recvbuff_ptr, count,
                                       dtype_, op_, comm_idx, stream_ptr,
                                       timeout_ms);
      },
      py::arg("sendbuff"), py::arg("recvbuff"), py::arg("count"),
      py::arg("dtype"), py::arg("op"), py::arg("comm_idx"), py::arg("stream"),
      py::arg("timeout_ms") = cosmos_reason1::get_default_timeout_ms(),
      py::call_guard<py::gil_scoped_release>(),
      R"pbdoc(
            Perform an NCCL allreduce operation.

            Args:
                sendbuff (void*): Tensor to send (must be CUDA and contiguous).
                recvbuff (void*): Tensor to receive into (must be CUDA and contiguous).
                count (int): Number of elements to reduce.
                dtype (ncclDataType_t): Data type of the tensor.
                op (ncclRedOp_t): Reduction operation.
                comm_idx (int): Communicator index.
                stream (cudaStream_t): CUDA stream to use for the operation.
                timeout_ms (int): Timeout value for the operation.
        )pbdoc");

  m.def(
      "nccl_alltoall",
      [](intptr_t sendbuff, intptr_t recvbuff, size_t total_size, int dtype,
         int64_t comm_idx, intptr_t stream, int64_t timeout_ms) {
        void* sendbuff_ptr = (void*)sendbuff;
        void* recvbuff_ptr = (void*)recvbuff;
        ncclDataType_t dtype_ = (ncclDataType_t)dtype;
        cudaStream_t stream_ptr = (cudaStream_t)stream;
        cosmos_reason1::nccl_alltoall(sendbuff_ptr, recvbuff_ptr, total_size,
                                      dtype_, comm_idx, stream_ptr, timeout_ms);
      },
      py::arg("sendbuff"), py::arg("recvbuff"), py::arg("total_size"),
      py::arg("dtype"), py::arg("comm_idx"), py::arg("stream"),
      py::arg("timeout_ms") = cosmos_reason1::get_default_timeout_ms(),
      py::call_guard<py::gil_scoped_release>(),
      R"pbdoc(
          Perform an NCCL alltoall operation.

          Args:
              sendbuff (void*): Tensor to send in alltoall (must be CUDA and contiguous).
              recvbuff (void*): Tensor to receive into in altoall (must be CUDA and contiguous).
              total_size (int): Total size of the tensor.
              dtype (ncclDataType_t): Data type of the tensor.
              comm_idx (int): Communicator index.
              stream (cudaStream_t): CUDA stream to use for the operation.
              timeout_ms (int): Timeout value for the operation.
      )pbdoc");
}
