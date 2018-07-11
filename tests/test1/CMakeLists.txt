cmake_minimum_required(VERSION 3.7)
project(test1)

set(CMAKE_CXX_STANDARD 17)

set(LOCH_SOURCE_DIR ../../src)
set(LOCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include/test1)

message(STATUS ${LOCH_SOURCE_DIR} " " ${LOCH_INCLUDE_DIR})

set(LOCH_SOURCE_FILES ${LOCH_SOURCE_DIR}/vec.cpp ${LOCH_SOURCE_DIR}/sim.cpp ${LOCH_SOURCE_DIR}/shader.cpp
        ${LOCH_SOURCE_DIR}/sim.cpp ${LOCH_SOURCE_DIR}/mass.cpp ${LOCH_SOURCE_DIR}/spring.cpp
        ${LOCH_SOURCE_DIR}/object.cpp ${LOCH_SOURCE_DIR}/graphics.cpp)

set(LOCH_HEADER_FILES ${LOCH_INCLUDE_DIR}/sim.h ${LOCH_INCLUDE_DIR}/mass.h ${LOCH_INCLUDE_DIR}/object.h
        ${LOCH_INCLUDE_DIR}/spring.h ${LOCH_INCLUDE_DIR}/vec.h ${LOCH_INCLUDE_DIR}/graphics.h ${LOCH_INCLUDE_DIR}/shader.h)

include_directories(${LOCH_INCLUDE_DIR})

find_package(OPENGL REQUIRED)
if (OPENGL_FOUND)
    message(STATUS "OPENGL FOUND")
    include_directories(${OPENGL_INCLUDE_DIRS})
    link_libraries(${OPENGL_LIBRARIES})
endif()

find_package(glfw3 REQUIRED)
if (glfw3_FOUND)
    message(STATUS "GLFW FOUND")
    include_directories(${glfw3_INCLUDE_DIRS})
    link_libraries(${glfw3_LIBRARIES})
endif()

find_package(GLEW REQUIRED)
if (GLEW_FOUND)
    message(STATUS "GLEW FOUND")
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
endif()

find_package(glm CONFIG REQUIRED)
if (glm_FOUND)
    message(STATUS "GLM FOUND")
    include_directories(${glm_INCLUDE_DIRS})
    link_libraries(${glm_LIBRARIES})
endif()

add_executable(test1.cpp ${LOCH_SOURCE_FILES} ${LOCH_HEADER_FILES})
target_compile_definitions(test1 PRIVATE GRAPHICS)

target_link_libraries(test1 PRIVATE glfw)
target_link_libraries(test1 PRIVATE glm)