cmake_minimum_required(VERSION 3.8)

project(pkrbot-sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(sim src/engine.cpp)
target_include_directories(sim PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(sim PRIVATE -O2 -Wall -Wextra)
