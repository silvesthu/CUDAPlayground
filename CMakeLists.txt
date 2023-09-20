cmake_minimum_required (VERSION 3.11)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Project
project ("CUDAPlayground")

# Add source to this project's executable.
add_executable (CUDAPlayground "CUDAPlayground.cu")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CUDAPlayground PROPERTY CXX_STANDARD 20)
endif()

# CUDA
enable_language(CUDA)
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
