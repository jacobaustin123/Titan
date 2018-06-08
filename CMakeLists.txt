cmake_minimum_required(VERSION 3.9)
project(Loch LANGUAGES CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)

set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++11
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14") # same thing, may be unnecessary

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src") # set SOURCE_DIR to src directory
include_directories(include) # include the include directory (can find headers there)

set(SOURCE_FILES src/vec.cu src/sim.cu src/sim.cu src/mass.cu src/spring.cu src/object.cu src/graphics.cpp src/common/shader.cpp) # add all of the .cu/.cpp files to SOURCE_FILES target

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

find_package(glfw3 CONFIG REQUIRED) # glfw
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

file(GLOB HEADERS "include/*.h") # add all headers to the project

add_executable(nographics src/main.cpp ${SOURCE_FILES} ${HEADERS}) # create nographics target
target_compile_features(nographics PUBLIC cxx_std_11)
set_target_properties(nographics PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # allows declarations and implementations to be separated
target_link_libraries(nographics PRIVATE cuda)



add_executable(graphics src/main.cpp ${SOURCE_FILES} ${HEADERS}) # create graphics target
target_compile_definitions(graphics PRIVATE GRAPHICS) # defines the GRAPHICS preprocessor variable

target_compile_features(graphics PUBLIC cxx_std_11) # same as above
set_target_properties(graphics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(graphics PRIVATE cuda)

target_link_libraries(graphics PRIVATE glfw) # link graphics target with the graphics libraries
target_link_libraries(graphics PRIVATE glm)

if ( WIN32 )
    target_link_libraries(graphics PRIVATE GLEW::GLEW)
else()
    target_link_libraries(graphics PRIVATE GLEW)
    target_link_libraries(graphics PRIVATE gl) # link OpenGL, seems to only matter on Mac OS
endif()