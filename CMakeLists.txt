cmake_minimum_required(VERSION 3.9)
project(Loch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")


set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(include)

set(SOURCE_FILES src/vec.cu src/sim.cu src/sim.cu src/mass.cu src/spring.cu src/object.cu src/graphics.cpp)

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA FOUND")
    include_directories(${CUDA_INCLUDE_DIRS})
    link_libraries(${CUDA_LIBRARIES})
else()
	message(STATUS "CUDA NOT FOUND")
endif()

find_package(OPENGL REQUIRED)
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

file(GLOB HEADERS "*.h")

add_executable(main src/main.cpp ${SOURCE_FILES} ${HEADERS})
target_compile_features(main PUBLIC cxx_std_11)

set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main cuda)

target_link_libraries(main PRIVATE glfw)
target_link_libraries(main PRIVATE glm)

if ( WIN32 )
    target_link_libraries(main PRIVATE GLEW::GLEW)
else()
    target_link_libraries(main PRIVATE GLEW)
    target_link_libraries(main PRIVATE gl)
endif()

install(TARGETS main DESTINATION ${CMAKE_BINARY_DIR})
