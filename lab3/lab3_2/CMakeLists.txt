cmake_minimum_required(VERSION 3.10)

project(lab3_2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lab32 lab3_2.cpp)

find_package(Threads REQUIRED)
target_link_libraries(lab3_2 PRIVATE Threads::Threads)