cmake_minimum_required(VERSION 3.7)
project(Loch)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(include)

set(SOURCE_FILES src/vec.cpp src/sim.cpp src/common/shader.cpp src/sim.cpp src/mass.cpp src/spring.cpp src/object.cpp)

file(GLOB HEADERS "*.h")

#add_subdirectory(src/dependencies/glfw)

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

find_package(glm REQUIRED)
if (glm_FOUND)
    message(STATUS "GLM FOUND")
    include_directories(${glm_INCLUDE_DIRS})
    link_libraries(${glm_LIBRARIES})
endif()

add_executable(main src/main.cpp ${SOURCE_FILES} ${HEADERS})
add_executable(debug src/mydebug.cpp ${SOURCE_FILES} ${HEADERS})

target_link_libraries(main glfw)
target_link_libraries(main gl)
target_link_libraries(main glew)
target_link_libraries(main glm)

target_link_libraries(debug glfw)
target_link_libraries(debug gl)
target_link_libraries(debug glew)
target_link_libraries(debug glm)

install(TARGETS main debug RUNTIME DESTINATION ${CMAKE_BINARY_DIR})
file(COPY src/shaders DESTINATION ${CMAKE_BINARY_DIR})