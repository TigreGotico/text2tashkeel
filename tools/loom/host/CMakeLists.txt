# Builds rawi_run against a loom.cpp checkout:
#   cmake -B build -DLOOM_CPP=/path/to/loom.cpp && cmake --build build -j
cmake_minimum_required(VERSION 3.16)
project(rawi_loom_host CXX)
set(CMAKE_CXX_STANDARD 17)
set(LOOM_CPP "" CACHE PATH "loom.cpp checkout with loom.run_bi_lstm_and_retain")
if(NOT LOOM_CPP)
    message(FATAL_ERROR "set -DLOOM_CPP=/path/to/loom.cpp")
endif()
set(LOOM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LOOM_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(${LOOM_CPP} loom)
add_executable(rawi_run rawi_run.cpp)
target_link_libraries(rawi_run PRIVATE loom::engine)
