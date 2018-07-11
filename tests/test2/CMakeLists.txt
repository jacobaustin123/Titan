cmake_minimum_required(VERSION 3.7)
project(test2)

set(CMAKE_CXX_STANDARD 17)

find_package(Loch CONFIG REQUIRED)
if (Loch_FOUND)
    message(STATUS "LOCH FOUND")
    include_directories(${Loch_INCLUDE_DIRS})
    link_libraries(${Loch_LIBRARIES})
endif()

add_executable(test2 ${CMAKE_CURRENT_LIST_DIR}/test2.cpp ${LOCH_SOURCE_FILES} ${LOCH_HEADER_FILES})

target_link_libraries(test2 ${Loch_LIBRARIES})