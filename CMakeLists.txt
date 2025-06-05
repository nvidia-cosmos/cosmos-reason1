cmake_minimum_required(VERSION 3.18)
project(cosmos_cpp)

cmake_policy(SET CMP0146 OLD)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
# Add CUDA flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(PYTHON_PATH "python" CACHE STRING "Python path")
set(COMMON_HEADER_DIRS "")
set(COMMON_LIB_DIRS "")

execute_process(
    COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch; print(torch.__version__,end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE TORCH_VERSION
)

execute_process(
    COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch; print(os.path.dirname(torch.__file__),end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE TORCH_DIR
)

if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Torch config Error.")
endif()

execute_process(
    COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch; print(os.path.dirname(os.path.dirname(torch.__file__)) + '/nvidia/nccl',end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE PY_NCCL_DIR
)
# Forward the include and lib directories to the build only if nccl lib and include dir exist
if (EXISTS ${PY_NCCL_DIR}/include AND EXISTS ${PY_NCCL_DIR}/lib)
    set(NCCL_INCLUDE_DIRS ${PY_NCCL_DIR}/include)
    # Find any libnccl.so files in the lib directory
    file(GLOB NCCL_LIBRARIES ${PY_NCCL_DIR}/lib/libnccl.so*)
    # Check if any matching libraries were found
    if (NCCL_LIBRARIES)
        list(GET NCCL_LIBRARIES 0 NCCL_LIB_PATH)  # Get the first match
        message(STATUS "Found NCCL library: ${NCCL_LIB_PATH}")
    else ()
        message(FATAL_ERROR "No libnccl.so files found in ${PY_NCCL_DIR}/lib")
    endif()
else()
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cosmos_reason1/cmake)
find_package(NCCL REQUIRED)
endif()

list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
message(STATUS "TORCH_DIR: ${TORCH_DIR}")
list(APPEND COMMON_HEADER_DIRS "${TORCH_DIR}/include")
list(APPEND COMMON_HEADER_DIRS "${TORCH_DIR}/include/torch/csrc/api/include")
list(APPEND COMMON_LIB_DIRS "${TORCH_DIR}/lib")
set(TORCH_LIBRARIES "torch;${TORCH_DIR}/lib/libc10.so;${TORCH_DIR}/lib/libtorch_python.so")

execute_process(
    COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig; print(sysconfig.get_python_inc());"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE PY_INCLUDE_DIR
)

if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Python config Error.")
endif()

list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})

if (USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

# Create the extension module
pybind11_add_module(_cpp
    cosmos_reason1/csrc/cosmos_reason1.cpp
)

include_directories(${COMMON_HEADER_DIRS} ${CUDA_INCLUDE_DIRS} ${NCCL_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/cosmos_reason1/csrc/include)

target_link_libraries(_cpp PRIVATE
    ${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${NCCL_LIBRARIES}
)
target_link_directories(_cpp PRIVATE ${COMMON_LIB_DIRS} ${NCCL_LIBRARIES})
