cmake_minimum_required(VERSION 3.25)

# Visual Studio generators: restore NuGet package references before build.
set(CMAKE_VS_NUGET_PACKAGE_RESTORE ON CACHE BOOL "Auto-restore NuGet packages for VS generators" FORCE)

# CMake's Visual Studio CUDA language support needs the CUDA MSBuild toolset
# before project(... CUDA) runs. Use CUDA_PATH when VS does not auto-detect it.
if(CMAKE_GENERATOR MATCHES "Visual Studio" AND NOT CMAKE_GENERATOR_TOOLSET AND DEFINED ENV{CUDA_PATH})
  file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" CUDAPLAYGROUND_CUDA_PATH)
  if(EXISTS "${CUDAPLAYGROUND_CUDA_PATH}/extras/visual_studio_integration/MSBuildExtensions")
    set(CMAKE_GENERATOR_TOOLSET
      "cuda=${CUDAPLAYGROUND_CUDA_PATH}"
      CACHE STRING
      "Visual Studio generator toolset"
      FORCE
    )
  endif()
endif()

option(CUDAPLAYGROUND_ALLOW_UNSUPPORTED_COMPILER
  "Pass -allow-unsupported-compiler to NVCC for newer host compilers"
  ON
)
if(CUDAPLAYGROUND_ALLOW_UNSUPPORTED_COMPILER)
  string(FIND "${CMAKE_CUDA_FLAGS}" "-allow-unsupported-compiler" _cuda_allow_unsupported_index)
  if(_cuda_allow_unsupported_index EQUAL -1)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
  endif()
endif()

project(CUDAPlayground LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDAPLAYGROUND_CUDA_ARCHITECTURES
  75
  CACHE STRING
  "CUDA architectures to build"
)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_package(CUDAToolkit REQUIRED)

set(CUDAPLAYGROUND_OPTIX_INCLUDE_HINTS)
foreach(_optix_env IN ITEMS OPTIX_PATH OPTIX_ROOT OPTIX_ROOT_DIR)
  if(DEFINED ENV{${_optix_env}})
    list(APPEND CUDAPLAYGROUND_OPTIX_INCLUDE_HINTS "$ENV{${_optix_env}}/include")
  endif()
endforeach()
file(GLOB CUDAPLAYGROUND_OPTIX_INSTALLED_INCLUDE_DIRS
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK */include"
)
list(APPEND CUDAPLAYGROUND_OPTIX_INCLUDE_HINTS ${CUDAPLAYGROUND_OPTIX_INSTALLED_INCLUDE_DIRS})

find_path(OPTIX_INCLUDE_DIR
  NAMES optix.h
  HINTS ${CUDAPLAYGROUND_OPTIX_INCLUDE_HINTS}
  DOC "Path to OptiX include directory"
)

if(NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
  message(FATAL_ERROR "OptiX headers were not found at: ${OPTIX_INCLUDE_DIR}")
endif()

set(CUDAPLAYGROUND_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/CUDAPlayground.cu"
)

set(CUDAPLAYGROUND_PROJECT_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/Common.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/CommonOptix.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/HelloCUDA.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/HelloOptix.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/HelloWMMA.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/HelloOptix.cu"
)

add_executable(CUDAPlayground
  ${CUDAPLAYGROUND_SOURCES}
  ${CUDAPLAYGROUND_PROJECT_FILES}
)
set_property(TARGET CUDAPlayground PROPERTY CUDA_ARCHITECTURES "${CUDAPLAYGROUND_CUDA_ARCHITECTURES}")

# Keep project-only files visible in VS Solution Explorer, but never build them.
set_source_files_properties(
  ${CUDAPLAYGROUND_PROJECT_FILES}
  PROPERTIES HEADER_FILE_ONLY ON
)

target_include_directories(CUDAPlayground PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${OPTIX_INCLUDE_DIR}"
)

target_compile_definitions(CUDAPlayground PRIVATE
  _CONSOLE
)

if(MSVC)
  target_compile_options(CUDAPlayground PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3>
    $<$<COMPILE_LANGUAGE:CXX>:/MP>
  )
endif()

target_link_libraries(CUDAPlayground PRIVATE
  CUDA::cudart_static
)

# Match the legacy project behavior: compile HelloOptix.cu as PTX.
set(HELLO_OPTIX_PTX "${CMAKE_CURRENT_BINARY_DIR}/HelloOptix.ptx")
set(HELLO_OPTIX_PTX_ARCHITECTURE "${CUDAPLAYGROUND_CUDA_ARCHITECTURES}")
list(GET HELLO_OPTIX_PTX_ARCHITECTURE 0 HELLO_OPTIX_PTX_ARCHITECTURE)
string(REGEX REPLACE "-.*$" "" HELLO_OPTIX_PTX_ARCHITECTURE "${HELLO_OPTIX_PTX_ARCHITECTURE}")
set(HELLO_OPTIX_NVCC_FLAGS
  --ptx
  --gpu-architecture=compute_${HELLO_OPTIX_PTX_ARCHITECTURE}
)
if(CUDAPLAYGROUND_ALLOW_UNSUPPORTED_COMPILER)
  list(APPEND HELLO_OPTIX_NVCC_FLAGS -allow-unsupported-compiler)
endif()
add_custom_command(
  OUTPUT "${HELLO_OPTIX_PTX}"
  COMMAND "${CMAKE_CUDA_COMPILER}"
    ${HELLO_OPTIX_NVCC_FLAGS}
    -I "${CMAKE_CURRENT_SOURCE_DIR}"
    -I "${OPTIX_INCLUDE_DIR}"
    -o "${HELLO_OPTIX_PTX}"
    "${CMAKE_CURRENT_SOURCE_DIR}/HelloOptix.cu"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/HelloOptix.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/Common.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/CommonOptix.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/HelloOptix.h"
  VERBATIM
)

add_custom_target(HelloOptixPtx ALL DEPENDS "${HELLO_OPTIX_PTX}")
add_dependencies(CUDAPlayground HelloOptixPtx)

# HelloOptix loads PTX from the process working directory.
add_custom_command(TARGET CUDAPlayground POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${HELLO_OPTIX_PTX}"
    "$<TARGET_FILE_DIR:CUDAPlayground>/HelloOptix.ptx"
  VERBATIM
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES
  ${CUDAPLAYGROUND_SOURCES}
  ${CUDAPLAYGROUND_PROJECT_FILES}
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT CUDAPlayground)
