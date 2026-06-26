cmake_minimum_required(VERSION 3.18)
project(heat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS program_options)

set(SRC ${CMAKE_SOURCE_DIR}/lab6.cpp)
set(PGC_FLAGS -fast -Minfo=all)

# ---------- GPU ----------
add_executable(heat_gpu ${SRC})
target_compile_options(heat_gpu PRIVATE ${PGC_FLAGS} -acc=gpu)
target_link_options(heat_gpu PRIVATE -acc=gpu)
target_link_libraries(heat_gpu PRIVATE Boost::program_options)

# ---------- Multicore CPU ----------
add_executable(heat_multicore ${SRC})
target_compile_options(heat_multicore PRIVATE ${PGC_FLAGS} -acc=multicore)
target_link_options(heat_multicore PRIVATE -acc=multicore)
target_link_libraries(heat_multicore PRIVATE Boost::program_options)

# ---------- Host (sequential) ----------
add_executable(heat_host ${SRC})
target_compile_options(heat_host PRIVATE ${PGC_FLAGS} -acc=host)
target_link_options(heat_host PRIVATE -acc=host)
target_link_libraries(heat_host PRIVATE Boost::program_options)

# ---------- GPU + NVTX (для профилирования Nsight Systems) ----------
set(NVTX_INCLUDE_DIR
    "/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include"
    CACHE PATH "Path to NVTX3 headers")

add_executable(heat_profile ${SRC})
target_compile_options(heat_profile PRIVATE ${PGC_FLAGS} -acc=gpu)
target_compile_definitions(heat_profile PRIVATE USE_NVTX)
target_include_directories(heat_profile PRIVATE ${NVTX_INCLUDE_DIR})
target_link_options(heat_profile PRIVATE -acc=gpu)
target_link_libraries(heat_profile PRIVATE Boost::program_options)