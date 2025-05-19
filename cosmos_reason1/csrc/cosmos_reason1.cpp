// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
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

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cosmos_reason1.h"

namespace py = pybind11;
static std::vector<ncclComm_t> shared_comms;

namespace cosmos_reason1 {

cudaStream_t getCurrentCUDAStream() {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    return stream;
}

torch::Tensor from_blob(int64_t data_ptr, std::vector<int64_t> shape, torch::Tensor sample) {
    auto options = torch::TensorOptions();
    options = options.dtype(sample.dtype()).device(sample.device());
    void* ptr = reinterpret_cast<void*>(data_ptr);
    return torch::from_blob(ptr, shape, options);
}

std::vector<int64_t> create_nccl_uid() {
    ncclUniqueId uid;
    NCCLCHECK(ncclGetUniqueId(&uid));
    std::vector<int64_t> result(128);
    for (int i = 0; i < 128; i++) {
        result[i] = uid.internal[i];
    }
    return result;
}

int64_t create_nccl_comm(std::vector<int64_t> uid_chars, int64_t rank, int64_t world_size) {
    ncclUniqueId uid;
    for (int i = 0; i < 128; i++) {
        uid.internal[i] = uid_chars[i];
    }
    static_assert(sizeof(ncclUniqueId) == 128, "ncclUniqueId size is not 128 bytes");
    ncclComm_t comm;

    auto functor = [&]() {
        NCCLCHECK(ncclCommInitRank(&comm, world_size, uid, rank));
    };
    if (PyGILState_Check()) {
        py::gil_scoped_release release;
        functor();
    } else {
        functor();
    }
    shared_comms.push_back(comm);
    return shared_comms.size() - 1;
}

ncclComm_t get_nccl_comm(int64_t idx) {
    COSMOS_CHECK_WITH_INFO(idx >= 0 && idx < shared_comms.size(), "Invalid NCCL communicator index");
    return shared_comms[idx];
}

int nccl_get_comm_count(int64_t comm_idx) {
    auto comm = get_nccl_comm(comm_idx);
    int nranks;
    NCCLCHECK(ncclCommCount(comm, &nranks));
    return nranks;
}

void nccl_broadcast(torch::Tensor tensor, int64_t rank, int64_t comm_idx) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(tensor.numel() > 0, "Tensor must have non-zero number of elements");

    int tensor_device_index = tensor.device().index();
    int current_device_index;
    COSMOS_CUDA_CHECK(cudaGetDevice(&current_device_index));
    TORCH_CHECK(current_device_index == tensor_device_index, 
                "Current CUDA device does not match tensor's device. Make sure to set the right device before using NCCL.");
                
    ncclComm_t comm = get_nccl_comm(comm_idx);
    size_t count = tensor.numel();
    ncclDataType_t dtype;

    switch (tensor.scalar_type()) {
        case torch::kFloat: dtype = ncclFloat; break;
        case torch::kHalf: dtype = ncclHalf; break;
        case torch::kBFloat16: dtype = ncclBfloat16; break;
        case torch::kDouble: dtype = ncclDouble; break;
        case torch::kInt: dtype = ncclInt; break;
        case torch::kLong: dtype = ncclInt64; break;
        case torch::kUInt8: dtype = ncclChar; break;
        case torch::kInt8: dtype = ncclChar; break;
        default:
            TORCH_CHECK(false, "Unsupported tensor dtype for NCCL broadcast");
    }

    void* data_ptr = tensor.data_ptr();

    TORCH_CHECK(data_ptr != nullptr, "Tensor data_ptr is null (tensor likely not materialized)");

    cudaStream_t stream = getCurrentCUDAStream();


    auto functor = [&]() {
        NCCLCHECK(ncclBroadcast(
            data_ptr,  // sendbuff
            data_ptr,  // recvbuff
            count,
            dtype,
            rank,
            comm,
            stream
        ));
    };

    if (PyGILState_Check()) {
        py::gil_scoped_release release;
        functor();
    } else {
        functor();
    }
}

void nccl_send(torch::Tensor tensor, int64_t peer, int64_t comm_idx) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(tensor.numel() > 0, "Tensor must have non-zero number of elements");

    int tensor_device_index = tensor.device().index();
    int current_device_index;
    COSMOS_CUDA_CHECK(cudaGetDevice(&current_device_index));
    TORCH_CHECK(current_device_index == tensor_device_index, 
                "Current CUDA device does not match tensor's device.");

    ncclComm_t comm = get_nccl_comm(comm_idx);
    size_t count = tensor.numel();
    ncclDataType_t dtype;

    switch (tensor.scalar_type()) {
        case torch::kFloat: dtype = ncclFloat; break;
        case torch::kHalf: dtype = ncclHalf; break;
        case torch::kBFloat16: dtype = ncclBfloat16; break;
        case torch::kDouble: dtype = ncclDouble; break;
        case torch::kInt: dtype = ncclInt; break;
        case torch::kLong: dtype = ncclInt64; break;
        case torch::kUInt8: dtype = ncclChar; break;
        case torch::kInt8: dtype = ncclChar; break;
        default:
            TORCH_CHECK(false, "Unsupported tensor dtype for NCCL send");
    }

    void* data_ptr = tensor.data_ptr();
    TORCH_CHECK(data_ptr != nullptr, "Tensor data_ptr is null");

    cudaStream_t stream = getCurrentCUDAStream();

    auto functor = [&]() {
        NCCLCHECK(ncclSend(
            data_ptr,
            count,
            dtype,
            peer,
            comm,
            stream
        ));
    };

    if (PyGILState_Check()) {
        py::gil_scoped_release release;
        functor();
    } else {
        functor();
    }
}

void nccl_recv(torch::Tensor tensor, int64_t peer, int64_t comm_idx) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(tensor.numel() > 0, "Tensor must have non-zero number of elements");

    int tensor_device_index = tensor.device().index();
    int current_device_index;
    COSMOS_CUDA_CHECK(cudaGetDevice(&current_device_index));
    TORCH_CHECK(current_device_index == tensor_device_index, 
                "Current CUDA device does not match tensor's device.");

    ncclComm_t comm = get_nccl_comm(comm_idx);
    size_t count = tensor.numel();
    ncclDataType_t dtype;

    switch (tensor.scalar_type()) {
        case torch::kFloat: dtype = ncclFloat; break;
        case torch::kHalf: dtype = ncclHalf; break;
        case torch::kBFloat16: dtype = ncclBfloat16; break;
        case torch::kDouble: dtype = ncclDouble; break;
        case torch::kInt: dtype = ncclInt; break;
        case torch::kLong: dtype = ncclInt64; break;
        case torch::kUInt8: dtype = ncclChar; break;
        case torch::kInt8: dtype = ncclChar; break;
        default:
            TORCH_CHECK(false, "Unsupported tensor dtype for NCCL recv");
    }

    void* data_ptr = tensor.data_ptr();
    TORCH_CHECK(data_ptr != nullptr, "Tensor data_ptr is null");

    cudaStream_t stream = getCurrentCUDAStream();

    auto functor = [&]() {
        NCCLCHECK(ncclRecv(
            data_ptr,
            count,
            dtype,
            peer,
            comm,
            stream
        ));
    };

    if (PyGILState_Check()) {
        py::gil_scoped_release release;
        functor();
    } else {
        functor();
    }
}

void nccl_allreduce(torch::Tensor sendbuff, torch::Tensor recvbuff, int64_t op, int64_t comm_idx) {
    TORCH_CHECK(sendbuff.is_cuda(), "Send Tensor must be a CUDA tensor");
    TORCH_CHECK(sendbuff.is_contiguous(), "Send Tensor must be contiguous");
    TORCH_CHECK(sendbuff.numel() > 0, "Send Tensor must have non-zero number of elements");
    TORCH_CHECK(recvbuff.is_cuda(), "Recv Tensor must be a CUDA tensor");
    TORCH_CHECK(recvbuff.is_contiguous(), "Recv Tensor must be contiguous");
    TORCH_CHECK(recvbuff.numel() > 0, "Recv Tensor must have non-zero number of elements");
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
    ncclDataType_t dtype;

    switch (sendbuff.scalar_type()) {
        case torch::kFloat: dtype = ncclFloat; break;
        case torch::kHalf: dtype = ncclHalf; break;
        case torch::kBFloat16: dtype = ncclBfloat16; break;
        case torch::kDouble: dtype = ncclDouble; break;
        case torch::kInt: dtype = ncclInt; break;
        case torch::kLong: dtype = ncclInt64; break;
        case torch::kUInt8: dtype = ncclChar; break;
        case torch::kInt8: dtype = ncclChar; break;
        default:
            TORCH_CHECK(false, "Unsupported tensor dtype for NCCL recv");
    }

    ncclRedOp_t red_op;
    switch (op) {
        case 0: red_op = ncclSum; break;
        case 1: red_op = ncclProd; break;
        case 2: red_op = ncclMax; break;
        case 3: red_op = ncclMin; break;
        case 4: red_op = ncclAvg; break;
        default:
            TORCH_CHECK(false, "Unsupported reduction operation for NCCL allreduce");
    }

    void* send_ptr = sendbuff.data_ptr();
    TORCH_CHECK(send_ptr != nullptr, "Send tensor data_ptr is null");

    void* recv_ptr = recvbuff.data_ptr();
    TORCH_CHECK(recv_ptr != nullptr, "Recv tensor data_ptr is null");

    cudaStream_t stream = getCurrentCUDAStream();

    auto functor = [&]() {
        NCCLCHECK(ncclAllReduce(
            send_ptr,
            recv_ptr,
            count,
            dtype,
            red_op,
            comm,
            stream
        ));
    };

    if (PyGILState_Check()) {
        py::gil_scoped_release release;
        functor();
    } else {
        functor();
    }
}

} // namespace cosmos_reason1

PYBIND11_MODULE(_cpp, m) {
    m.doc() = "Cosmos C++/CUDA extension";
    m.def("create_nccl_comm", &cosmos_reason1::create_nccl_comm, "Create a NCCL communicator");
    // m.def("get_nccl_comm", &cosmos_reason1::get_nccl_comm, "Get a NCCL communicator");
    m.def("from_blob", &cosmos_reason1::from_blob, "Create a tensor from a blob");
    m.def("create_nccl_uid", &cosmos_reason1::create_nccl_uid, "Create a NCCL unique ID");
    m.def("get_nccl_comm_count", &cosmos_reason1::nccl_get_comm_count, 
        py::arg("comm_idx"),
        "Get the number of ranks in the NCCL communicator");
    m.def("nccl_broadcast", &cosmos_reason1::nccl_broadcast,
        py::arg("tensor"),
        py::arg("rank"),
        py::arg("comm_idx"),
        R"pbdoc(
            Perform an NCCL broadcast on the given NCCL group.

            Args:
                tensor (torch.Tensor): Tensor to broadcast (must be on CUDA).
                root (int): Root rank in the communicator.
                comm_idx (int): Index of the communicator (created by `create_nccl_comm`).
        )pbdoc");

    m.def("nccl_send", &cosmos_reason1::nccl_send,
        py::arg("tensor"),
        py::arg("peer"),
        py::arg("comm_idx"),
        R"pbdoc(
            Perform an NCCL point-to-point send operation.

            Args:
                tensor (torch.Tensor): Tensor to send (must be CUDA and contiguous).
                peer (int): Rank to send to.
                comm_idx (int): Communicator index.
        )pbdoc");

    m.def("nccl_recv", &cosmos_reason1::nccl_recv,
        py::arg("tensor"),
        py::arg("peer"),
        py::arg("comm_idx"),
        R"pbdoc(
            Perform an NCCL point-to-point recv operation.

            Args:
                tensor (torch.Tensor): Tensor to receive into (must be CUDA and contiguous).
                peer (int): Rank to receive from.
                comm_idx (int): Communicator index.
        )pbdoc");

    m.def("nccl_allreduce", &cosmos_reason1::nccl_allreduce,
        py::arg("sendbuff"),
        py::arg("recvbuff"),
        py::arg("op"),
        py::arg("comm_idx"),
        R"pbdoc(
            Perform an NCCL allreduce operation.

            Args:
                sendbuff (torch.Tensor): Tensor to send (must be CUDA and contiguous).
                recvbuff (torch.Tensor): Tensor to receive into (must be CUDA and contiguous).
                op (int): Reduction operation (0: sum, 1: prod, 2: max, 3: min, 4: avg).
                comm_idx (int): Communicator index.
        )pbdoc");
} 
