cmake_minimum_required(VERSION 3.11)
project(Loch LANGUAGES CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)

set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++11

set(LOCH_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(LOCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/Loch)

set(LOCH_SOURCE_FILES ${LOCH_SOURCE_DIR}/vec.cu ${LOCH_SOURCE_DIR}/sim.cu ${LOCH_SOURCE_DIR}/shader.cpp
        ${LOCH_SOURCE_DIR}/mass.cu ${LOCH_SOURCE_DIR}/spring.cu
        ${LOCH_SOURCE_DIR}/object.cu ${LOCH_SOURCE_DIR}/graphics.cpp)

set(LOCH_HEADER_FILES ${LOCH_INCLUDE_DIR}/sim.h ${LOCH_INCLUDE_DIR}/mass.h ${LOCH_INCLUDE_DIR}/object.h
        ${LOCH_INCLUDE_DIR}/spring.h ${LOCH_INCLUDE_DIR}/vec.h ${LOCH_INCLUDE_DIR}/graphics.h ${LOCH_INCLUDE_DIR}/shader.h)

add_executable(Loch main.cpp ${LOCH_SOURCE_FILES} ${LOCH_HEADER_FILES})

set_target_properties(Loch PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # needed for library compilation
target_include_directories(Loch PRIVATE include/Loch)

target_compile_definitions(Loch PUBLIC GRAPHICS)
target_compile_definitions(Loch PUBLIC CONSTRAINTS) # defines the CONSTRAINTS preprocessor variable (enables local constraints)

find_package(CUDA REQUIRED) # find and include CUDA
if (CUDA_FOUND)
    message(STATUS "CUDA FOUND")
    target_include_directories(Loch PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(Loch PRIVATE cuda) # cudart
else()
    message(STATUS "CUDA NOT FOUND")
endif()

message(STATUS ${CUDA_LIBRARIES})

find_package(assimp REQUIRED) # find and include OpenGL
if (assimp_FOUND)
    message(STATUS "ASSIMP FOUND")
    target_include_directories(Loch PRIVATE ${assimp_INCLUDE_DIRS})
    target_link_libraries(Loch PRIVATE ${assimp_LIBRARIES})
endif()

find_package(OPENGL REQUIRED) # find and include OpenGL
if (OPENGL_FOUND)
    message(STATUS "OPENGL FOUND")
    target_include_directories(Loch PRIVATE ${OPENGL_INCLUDE_DIRS})
    target_link_libraries(Loch PRIVATE ${OPENGL_LIBRARIES})
endif()

find_package(glfw3 CONFIG REQUIRED)
if (glfw3_FOUND)
    message(STATUS "GLFW FOUND")
    target_include_directories(Loch PRIVATE ${glfw3_INCLUDE_DIRS})
    target_link_libraries(Loch PRIVATE ${glfw3_LIBRARIES})
endif()

find_package(GLEW REQUIRED) # GLEW
if (GLEW_FOUND)
    message(STATUS "GLEW FOUND")
    target_include_directories(Loch PRIVATE ${GLEW_INCLUDE_DIRS})
    target_link_libraries(Loch PRIVATE ${GLEW_LIBRARIES} glfw)
endif()

find_package(glm CONFIG REQUIRED) # glm
if (glm_FOUND)
    message(STATUS "GLM FOUND")
    target_include_directories(Loch PRIVATE ${glm_INCLUDE_DIRS})
    target_link_libraries(Loch PRIVATE ${glm_LIBRARIES} glm)
endif()