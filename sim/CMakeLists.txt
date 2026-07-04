cmake_minimum_required(VERSION 3.18)

project(pkrbot-sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Shared include dirs and compile options for all simulator targets.
add_library(sim_core INTERFACE)
target_include_directories(sim_core INTERFACE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(sim_core INTERFACE -O2 -Wall -Wextra)

find_package(Threads REQUIRED)

# libtorch for the train target's inference engine (torch_engine.cpp). Point
# CMAKE_PREFIX_PATH at the pip-installed torch, e.g.
#   cmake -DCMAKE_PREFIX_PATH=$(.venv/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)') ..
find_package(Torch REQUIRED)
separate_arguments(TORCH_CXX_FLAGS_LIST UNIX_COMMAND "${TORCH_CXX_FLAGS}")

add_executable(sim src/engine.cpp)
target_link_libraries(sim PRIVATE sim_core)

add_executable(train src/train.cpp src/traverse.cpp src/inference.cpp src/torch_engine.cpp)
target_compile_options(train PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_link_libraries(train PRIVATE sim_core Threads::Threads ${TORCH_LIBRARIES})

# pip-wheel libtorch locates its bundled NVIDIA deps (nccl, cudnn, ...) via
# $ORIGIN rpaths, which ld does not expand when checking shared-library
# undefined symbols at link time; hand it those directories explicitly.
get_filename_component(TORCH_SITE_PACKAGES "${Torch_DIR}/../../../.." ABSOLUTE)
file(GLOB TORCH_NVIDIA_LIB_DIRS "${TORCH_SITE_PACKAGES}/nvidia/*/lib")
foreach(dir ${TORCH_NVIDIA_LIB_DIRS})
  target_link_options(train PRIVATE "-Wl,-rpath-link,${dir}")
endforeach()

add_executable(batch_test src/batch_test.cpp src/inference.cpp)
target_link_libraries(batch_test PRIVATE sim_core Threads::Threads)
