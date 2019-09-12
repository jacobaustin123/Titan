cmake_minimum_required(VERSION 3.9)
project(test3 LANGUAGES CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)

add_executable(test3 ${CMAKE_CURRENT_LIST_DIR}/test3.cpp)

find_package(Titan CONFIG REQUIRED)
target_link_libraries(test3 PRIVATE Titan)

