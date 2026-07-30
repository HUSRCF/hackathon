cmake_minimum_required(VERSION 3.23)
project(tripharm_hip LANGUAGES CXX HIP)

get_filename_component(_hip_compiler_bin "${CMAKE_HIP_COMPILER}" DIRECTORY)
get_filename_component(_hip_llvm_root "${_hip_compiler_bin}" DIRECTORY)
get_filename_component(_hip_runtime_lib "${_hip_llvm_root}" DIRECTORY)

add_executable(tripharm_hip_benchmark tripharm_hip.cpp)
add_executable(tripharm_hip_query tripharm_hip_query.cpp)
set_source_files_properties(tripharm_hip.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(tripharm_hip_query.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(
  tripharm_hip_benchmark
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    HIP_STANDARD 17
    BUILD_RPATH "${_hip_runtime_lib}"
    INSTALL_RPATH "${_hip_runtime_lib}"
)
target_compile_options(tripharm_hip_benchmark PRIVATE -O3)
set_target_properties(
  tripharm_hip_query
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    HIP_STANDARD 17
    BUILD_RPATH "${_hip_runtime_lib}"
    INSTALL_RPATH "${_hip_runtime_lib}"
)
target_compile_options(tripharm_hip_query PRIVATE -O3)
