cmake_minimum_required(VERSION 3.7)
project(Loch)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(include)

set(SOURCE_FILES src/vec.cpp src/sim.cpp src/common/shader.cpp src/sim.cpp src/mass.cpp src/spring.cpp src/object.cpp src/graphics.cpp include/sim.h include/mass.h include/object.h include/spring.h include/vec.h include/graphics.h)

#find_package(OPENGL REQUIRED)
#if (OPENGL_FOUND)
#    message(STATUS "OPENGL FOUND")
#    include_directories(${OPENGL_INCLUDE_DIRS})
#    link_libraries(${OPENGL_LIBRARIES})
#endif()
#
#find_package(glfw3 CONFIG REQUIRED)
#if (glfw3_FOUND)
#    message(STATUS "GLFW FOUND")
#    include_directories(${glfw3_INCLUDE_DIRS})
#    link_libraries(${glfw3_LIBRARIES})
#endif()
#
#find_package(GLEW REQUIRED)
#if (GLEW_FOUND)
#    message(STATUS "GLEW FOUND")
#    include_directories(${GLEW_INCLUDE_DIRS})
#    link_libraries(${GLEW_LIBRARIES})
#endif()
#
#find_package(glm CONFIG REQUIRED)
#if (glm_FOUND)
#    message(STATUS "GLM FOUND")
#    include_directories(${glm_INCLUDE_DIRS})
#    link_libraries(${glm_LIBRARIES})
#endif()

#file(COPY ${SOURCE_DIR}/shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_library(nographics ${SOURCE_FILES} ${HEADERS})

set(Loch_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
#set(Loch_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/nographics.a)