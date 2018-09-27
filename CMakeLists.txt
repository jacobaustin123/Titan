#set(PYTHON_EXECUTABLE C:/Users/rcorr/Anaconda3/python.exe)
cmake_minimum_required(VERSION 3.9)
project(Loch LANGUAGES CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++11
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14") # same thing, may be unnecessary
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src") # set SOURCE_DIR to src directory
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include") # set SOURCE_DIR to src directory

include_directories(include) # include the include directory (can find headers there)

#SET CPP PLOCH PROJECT
set(LOCH_SOURCE_FILES src/object.cu src/graphics.cpp src/mass.cu src/sim.cu src/spring.cu src/vec.cu src/common/shader.cpp)
set(LOCH_HEADER_FILES include/object.h include/graphics.h include/mass.h include/sim.h include/spring.h include/vec.h)
set(LOCH_FILES ${LOCH_HEADER_FILES} ${LOCH_SOURCE_FILES})
set(LOCH_INCLUDE_DIR include)
#SET PYTHON BINDING DIRECTORIES
set(LOCH_BIND_SOURCE_FILES  bind/src/pysimulation.cpp bind/src/pymass.cpp bind/src/pyspring.cpp bind/src/bind.cpp
        bind/src/bind_mass.cpp  bind/src/bind_sim.cpp bind/src/bind_spring.cpp bind/src/pyspring.cpp
        bind/src/pymass.cpp)
set(LOCH_BIND_HEADER_FILES bind/include/pysimulation.h bind/include/pyspring.h bind/include/pymass.h)
set(LOCH_BIND_FILES ${LOCH_BIND_HEADER_FILES} ${LOCH_BIND_SOURCE_FILES})
set(LOCH_BIND_INCLUDE_DIR bind/include)

#IMPORT PACKAGES
find_package(CUDA REQUIRED) # find and include CUDA
if (CUDA_FOUND)
    message(STATUS "CUDA FOUND")
    list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
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

if (WIN32)
    find_package(glfw3 CONFIG REQUIRED)
    if (glfw3_FOUND)
        message(STATUS "GLFW FOUND")
        include_directories(${glfw3_INCLUDE_DIRS})
        link_libraries(${glfw3_LIBRARIES})
    endif()
else()
    find_package(sdl2 CONFIG REQUIRED)
    if (sdl2_FOUND)
        message(STATUS "sdl2 FOUND")
        include_directories(${sdl2_INCLUDE_DIRS})
        link_libraries(${sdl2_LIBRARIES})
    endif()
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

find_package(pybind11 REQUIRED)  # or add_subdirectory(pybind11)
if (pybind11_FOUND)
    message(STATUS "pybind11 FOUND")
else()
    message(STATUS "pybind11 NOT FOUND")
endif()

#FIND PYTHON: ONLY NEEDED FOR DEPENDENT BINDINGS APPROACH(SRC CPP CODE IS MODIFIED TO OUTPUT PYTHON OBJECTS)
#DECISION WAS MADE TO MAKE THE CPP MODULE AND PYTHON BINDINGS COMPLETELY INDEPENDENT FROM EACH OTHER

#find_package(PythonLibs REQUIRED)
#if (PythonLibs_FOUND)
#    message(STATUS "Python FOUND")
#else()
#    message(STATUS "Python NOT FOUND")
#endif()


#Create Library with .cu files
add_library(cu_lib ${LOCH_FILES})
target_include_directories(cu_lib PRIVATE ${LOCH_INCLUDE_DIR}} ${CUDA_INCLUDE_DIRS} )
set_target_properties(cu_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cu_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cu_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_definitions(cu_lib PRIVATE CONSTRAINTS) # defines the CONSTRAINTS preprocessor variable (enables local constraints)
#target_compile_definitions(cu_lib PRIVATE GRAPHICS) # defines the GRAPHICS preprocessor variable

target_compile_features(cu_lib PUBLIC cxx_std_11)
target_link_libraries(cu_lib PRIVATE cuda cudart glm)

if ( WIN32 ) # use GLFW on Windows
    target_link_libraries(cu_lib PRIVATE GLEW::GLEW)
    target_link_libraries(cu_lib PRIVATE glfw)
endif()

#PYBIND11
#CREATE PYTHON BINDINGS
pybind11_add_module(bindloch ${LOCH_BIND_FILES} ${LOCH_HEADER_FILES})

#Add linking dependencies to the binding library
target_include_directories(bindloch PRIVATE ${LOCH_BIND_INCLUDE_DIR}
        ${LOCH_INCLUDE_DIR})
add_dependencies(bindloch cu_lib)
target_link_libraries(bindloch PRIVATE cu_lib)


