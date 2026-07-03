cmake_minimum_required(VERSION 3.8)

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

add_executable(sim src/engine.cpp)
target_link_libraries(sim PRIVATE sim_core)

add_executable(train src/train.cpp src/traverse.cpp src/inference.cpp)
target_link_libraries(train PRIVATE sim_core Threads::Threads)

add_executable(batch_test src/batch_test.cpp src/inference.cpp)
target_link_libraries(batch_test PRIVATE sim_core Threads::Threads)
