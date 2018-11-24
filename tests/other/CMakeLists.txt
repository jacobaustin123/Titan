cmake_minimum_required(VERSION 3.9)
project(test1 LANGUAGES CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)

set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++11
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14") # same thing, may be unnecessary

set(TITAN_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../src)
set(TITAN_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../include/Titan)

set(TITAN_SOURCE_FILES ${TITAN_SOURCE_DIR}/vec.cu ${TITAN_SOURCE_DIR}/sim.cu ${TITAN_SOURCE_DIR}/shader.cpp
        ${TITAN_SOURCE_DIR}/sim.cu ${TITAN_SOURCE_DIR}/mass.cu ${TITAN_SOURCE_DIR}/spring.cu
        ${TITAN_SOURCE_DIR}/object.cu ${TITAN_SOURCE_DIR}/graphics.cpp)

set(TITAN_HEADER_FILES ${TITAN_INCLUDE_DIR}/sim.h ${TITAN_INCLUDE_DIR}/mass.h ${TITAN_INCLUDE_DIR}/object.h
        ${TITAN_INCLUDE_DIR}/spring.h ${TITAN_INCLUDE_DIR}/vec.h ${TITAN_INCLUDE_DIR}/graphics.h ${TITAN_INCLUDE_DIR}/shader.h)

include_directories(${TITAN_INCLUDE_DIR})

find_package(CUDA REQUIRED) # find and include CUDA
if (CUDA_FOUND)
    message(STATUS "CUDA FOUND")
    include_directories(${CUDA_INCLUDE_DIRS})
    link_libraries(${CUDA_LIBRARIES})
else()
    message(STATUS "CUDA NOT FOUND")
endif()

find_package(OPENGL REQUIRED) # find and include OpenGL
if (OPENGL_FOUND)
    message(STATUS "OPENGL FOUND")
    include_directories(${OPENGL_INCLUDE_DIRS})
    link_libraries(${OPENGL_LIBRARIES})
endif()

find_package(glfw3 CONFIG REQUIRED)
if (glfw3_FOUND)
    message(STATUS "GLFW FOUND")
    include_directories(${glfw3_INCLUDE_DIRS})
    link_libraries(${glfw3_LIBRARIES})
endif()


find_package(GLEW REQUIRED) # GLEW
if (GLEW_FOUND)
    message(STATUS "GLEW FOUND")
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
endif()

find_package(glm CONFIG REQUIRED) # glm
if (glm_FOUND)
    message(STATUS "GLM FOUND")
    include_directories(${glm_INCLUDE_DIRS})
    link_libraries(${glm_LIBRARIES})
endif()

add_executable(test1 ${CMAKE_CURRENT_LIST_DIR}/test1.cpp ${TITAN_SOURCE_FILES} ${TITAN_HEADER_FILES})
target_compile_definitions(test1 PRIVATE GRAPHICS)
target_compile_definitions(test1 PRIVATE CONSTRAINTS) # defines the CONSTRAINTS preprocessor variable (enables local constraints)

target_compile_features(test1 PUBLIC cxx_std_11) # same as above
set_target_properties(test1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(test1 PRIVATE cuda)
target_link_libraries(test1 PRIVATE glm)
target_link_libraries(test1 PRIVATE glfw)

if ( WIN32 )
    target_link_libraries(test1 PRIVATE GLEW::GLEW)
else()
    target_link_libraries(test1 PRIVATE GLEW)
endif()