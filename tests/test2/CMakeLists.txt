cmake_minimum_required(VERSION 3.9)
project(test2 LANGUAGES CXX) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)

find_package(Titan CONFIG REQUIRED)
target_link_libraries(test2 PRIVATE Titan)

add_executable(test2 ${CMAKE_CURRENT_LIST_DIR}/test2.cpp)

