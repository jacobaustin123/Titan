cmake_minimum_required(VERSION 3.9)
project(Loch LANGUAGES CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)

set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++11
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14") # same thing, may be unnecessary

set(LOCH_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(LOCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/Loch)

set(LOCH_SOURCE_FILES ${LOCH_SOURCE_DIR}/vec.cu ${LOCH_SOURCE_DIR}/sim.cu ${LOCH_SOURCE_DIR}/shader.cpp
        ${LOCH_SOURCE_DIR}/mass.cu ${LOCH_SOURCE_DIR}/spring.cu ${LOCH_SOURCE_DIR}/object.cu
        ${LOCH_SOURCE_DIR}/graphics.cpp)

set(LOCH_HEADER_FILES ${LOCH_INCLUDE_DIR}/sim.h ${LOCH_INCLUDE_DIR}/mass.h ${LOCH_INCLUDE_DIR}/object.h
        ${LOCH_INCLUDE_DIR}/spring.h ${LOCH_INCLUDE_DIR}/vec.h ${LOCH_INCLUDE_DIR}/graphics.h ${LOCH_INCLUDE_DIR}/shader.h)

include_directories(${LOCH_INCLUDE_DIR})

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

add_library(Loch ${LOCH_SOURCE_FILES} ${LOCH_HEADER_FILES})

target_compile_definitions(Loch PRIVATE GRAPHICS)
target_compile_definitions(Loch PRIVATE CONSTRAINTS) # defines the CONSTRAINTS preprocessor variable (enables local constraints)

#target_compile_features(Loch PUBLIC cxx_std_11) # same as above
set_target_properties(Loch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(Loch PRIVATE cuda)
#target_link_libraries(Loch PRIVATE glm)
#target_link_libraries(Loch PRIVATE glfw)
#
#if ( WIN32 )
#    target_link_libraries(Loch PRIVATE GLEW::GLEW)
#else()
#    target_link_libraries(Loch PRIVATE GLEW)
#endif()

#if (WIN32)
#    set(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/Common\ Files)
#endif()

install(DIRECTORY ${LOCH_INCLUDE_DIR} DESTINATION include/Loch)
install(TARGETS Loch DESTINATION lib)

set(Loch_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include/Loch)
set(Loch_LIBRARIES ${CMAKE_INSTALL_PREFIX}/lib/Loch)

configure_file(cmake/LochConfig.cmake.in
        ${CMAKE_INSTALL_PREFIX}/lib/cmake/Loch/LochConfig.cmake @ONLY)

#add_subdirectory(tests/test1)
#add_subdirectory(tests/test2)