cmake_minimum_required(VERSION 3.7)
project(CUDA)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
configure_file(main.cpp.in ${SOURCE_DIR}/main.cpp)

set(SOURCE_FILES main.cpp cube.cpp vec.cpp common/shader.cpp)

if (WIN32)
    include_directories(dependencies/glew-2.1.0/include) #  /usr/local/Cellar/glew/2.1.0/include dependencies/glfw/include # /usr/local/include
endif(WIN32)

find_package (glfw3 REQUIRED)
include_directories(${glfw3_INCLUDE_DIR})

find_package (OPENGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

find_package (glew REQUIRED)
include_directories(${glew_INCLUDE_DIR})

add_executable(main ${SOURCE_FILES})

target_link_libraries(main ${glfw3_LIBRARY})
target_link_libraries(main glfw)
target_link_libraries(main ${OPENGL_LIBRARY})
target_link_libraries(main glew)

find_package (glm REQUIRED)
include_directories(${glm_INCLUDE_DIR})
target_link_libraries(main ${glm_LIBRARY})