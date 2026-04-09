cmake_minimum_required(VERSION 3.10)

project(lab3_1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lab31 lab3_1.cpp)

find_package(Threads REQUIRED)
target_link_libraries(lab31 PRIVATE Threads::Threads)