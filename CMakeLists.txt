cmake_minimum_required(VERSION 3.7)
project(Loch)

set(CMAKE_CXX_STANDARD 17)

set(Loch_DIR ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Loch CONFIG REQUIRED)
if (Loch_FOUND)
    message(STATUS "LOCH FOUND")
    include_directories(${Loch_INCLUDE_DIRS})
    link_libraries(${Loch_LIBRARIES})
endif()

add_executable(main src/main.cpp ${SOURCE_FILES} ${HEADERS})
target_link_libraries(main graphics)
target_compile_definitions(main PRIVATE GRAPHICS) # todo fix this