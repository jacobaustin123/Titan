cmake_minimum_required(VERSION 3.9)
project(test2 LANGUAGES CXX) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)

find_package(Loch)
if (Loch_FOUND)
    message(STATUS "Loch FOUND")
    include_directories(${Loch_INCLUDE_DIRS})
    link_libraries(${Loch_LIBRARIES})
else()
    message(STATUS "LOCH NOT FOUND")
endif()

add_executable(test2 test2.cpp)
target_link_libraries(test2 ${Loch_LIBRARIES})