cmake_minimum_required(VERSION 3.7)
project(Loch)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(include)

#find_package(boost REQUIRED)
#if (boost_FOUND)
#    message(STATUS "BOOST FOUND")
#    include_directories(${boost_INCLUDE_DIRS})
#    link_libraries(${boost_LIBRARIES})
#endif()

set(SOURCE_FILES src/vec.cpp src/sim.cpp src/sim.cpp src/mass.cpp src/spring.cpp src/object.cpp)

file(GLOB HEADERS "*.h")
add_executable(main src/main.cpp ${SOURCE_FILES} ${HEADERS})
#target_link_libraries(main boost)

install(TARGETS main DESTINATION ${CMAKE_BINARY_DIR})
