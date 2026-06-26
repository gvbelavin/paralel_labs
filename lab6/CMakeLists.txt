cmake_minimum_required(VERSION 3.18)
project(Task6HeatEquation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Режим OpenACC: host / gpu / multicore
set(ACC_TARGET "host" CACHE STRING "OpenACC target: host, gpu, or multicore")
set_property(CACHE ACC_TARGET PROPERTY STRINGS host gpu multicore)

find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(task6 lab6.cpp)

target_include_directories(task6 PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(task6 PRIVATE Boost::program_options)

target_compile_options(task6 PRIVATE
    -O3
    -acc
    -acc=${ACC_TARGET}
    -Minfo=all
)

target_link_options(task6 PRIVATE
    -acc
    -acc=${ACC_TARGET}
)