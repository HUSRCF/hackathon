cmake_minimum_required(VERSION 3.23)
project(tripharm_hip LANGUAGES CXX HIP)

add_executable(tripharm_hip_benchmark tripharm_hip.cpp)
set_source_files_properties(tripharm_hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(
  tripharm_hip_benchmark
  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES HIP_STANDARD 17
)
target_compile_options(tripharm_hip_benchmark PRIVATE -O3)
