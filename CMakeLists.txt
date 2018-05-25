cmake_minimum_required(VERSION 3.7)
project(Loch)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(include)

set(SOURCE_FILES src/vec.cu src/sim.cu src/sim.cu src/mass.cpp src/spring.cpp src/object.cpp)

file(GLOB HEADERS "*.h")
add_executable(main src/main.cpp ${SOURCE_FILES} ${HEADERS})

install(TARGETS main DESTINATION ${CMAKE_BINARY_DIR})
