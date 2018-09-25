cmake_minimum_required(VERSION 3.11)
project(Loch LANGUAGES C CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)

set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++11

set(LOCH_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(LOCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/Loch)

set(LOCH_SOURCE_FILES ${LOCH_SOURCE_DIR}/vec.cu ${LOCH_SOURCE_DIR}/sim.cu ${LOCH_SOURCE_DIR}/shader.cpp
        ${LOCH_SOURCE_DIR}/mass.cu ${LOCH_SOURCE_DIR}/spring.cu
        ${LOCH_SOURCE_DIR}/object.cu ${LOCH_SOURCE_DIR}/graphics.cpp)

set(LOCH_HEADER_FILES ${LOCH_INCLUDE_DIR}/sim.h ${LOCH_INCLUDE_DIR}/mass.h ${LOCH_INCLUDE_DIR}/object.h
        ${LOCH_INCLUDE_DIR}/spring.h ${LOCH_INCLUDE_DIR}/vec.h ${LOCH_INCLUDE_DIR}/graphics.h ${LOCH_INCLUDE_DIR}/shader.h
        ${LOCH_INCLUDE_DIR}/stlparser.h)

if (LOCH_SHARED_BUILD)
    set(BUILD_SHARED_LIBS ON)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    set(POSITION_INDEPENDENT_CODE ON)
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(POSITION_INDEPENDENT_CODE ON)

add_library(Loch ${LOCH_SOURCE_FILES} ${LOCH_HEADER_FILES})

set_target_properties(Loch PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # needed for library compilation
set_target_properties(Loch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_definitions(Loch PUBLIC GRAPHICS)
target_compile_definitions(Loch PUBLIC CONSTRAINTS) # defines the CONSTRAINTS preprocessor variable (enables local constraints)

if (NOT LOCH_SHARED_BUILD)
    target_compile_definitions(Loch PUBLIC _MWAITXINTRIN_H_INCLUDED)
endif()

find_package(CUDA REQUIRED) # find and include CUDA
if (CUDA_FOUND)
    message(STATUS "CUDA FOUND")
    target_include_directories(Loch PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(Loch PRIVATE cuda) # cudart
else()
    message(STATUS "CUDA NOT FOUND")
endif()

find_package(OpenGL REQUIRED) # find and include OpenGL
if (OpenGL_FOUND)
    message(STATUS "OPENGL FOUND")
    target_include_directories(Loch PRIVATE ${OpenGL_INCLUDE_DIRS})

    if (WIN32)
        target_link_libraries(Loch PRIVATE ${OpenGL_LIBRARIES})
    else()
        target_link_libraries(Loch PRIVATE ${OpenGL_LIBRARIES} GL)
    endif()
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

target_include_directories(Loch PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Loch>
        $<INSTALL_INTERFACE:include>)

set_target_properties(Loch PROPERTIES PUBLIC_HEADER "${LOCH_HEADER_FILES}")

install(TARGETS Loch
        EXPORT LochTargets
        PUBLIC_HEADER DESTINATION include/Loch
        LIBRARY DESTINATION bin
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        )

if (LOCH_SHARED_BUILD)
    install(EXPORT LochTargets
            FILE LochConfig.cmake
            DESTINATION share/loch)
else()
    install(EXPORT LochTargets
            FILE LochTargets.cmake
            DESTINATION share/loch)

    configure_file(cmake/LochConfig.cmake.in2
            ${CMAKE_INSTALL_PREFIX}/share/loch/LochConfig.cmake @ONLY)
endif()