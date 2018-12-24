#cmake_minimum_required(VERSION 3.10)
#project(Titan LANGUAGES C CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)
#
#set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++11
#
#set(TITAN_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
#set(TITAN_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/Titan)
#
#set(TITAN_SOURCE_FILES ${TITAN_SOURCE_DIR}/vec.cu ${TITAN_SOURCE_DIR}/sim.cu ${TITAN_SOURCE_DIR}/shader.cpp
#        ${TITAN_SOURCE_DIR}/mass.cu ${TITAN_SOURCE_DIR}/spring.cu
#        ${TITAN_SOURCE_DIR}/object.cu ${TITAN_SOURCE_DIR}/graphics.cpp)
#
#set(TITAN_HEADER_FILES ${TITAN_INCLUDE_DIR}/sim.h ${TITAN_INCLUDE_DIR}/mass.h ${TITAN_INCLUDE_DIR}/object.h
#        ${TITAN_INCLUDE_DIR}/spring.h ${TITAN_INCLUDE_DIR}/vec.h ${TITAN_INCLUDE_DIR}/graphics.h ${TITAN_INCLUDE_DIR}/shader.h
#        ${TITAN_INCLUDE_DIR}/stlparser.h)
#
#if (TITAN_SHARED_BUILD)
#    set(BUILD_SHARED_LIBS ON)
#    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
#    set(POSITION_INDEPENDENT_CODE ON)
#endif()
#
#set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
#set(POSITION_INDEPENDENT_CODE ON)
#
#add_library(Titan ${TITAN_SOURCE_FILES} ${TITAN_HEADER_FILES})
#
#set_target_properties(Titan PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # needed for library compilation
#set_target_properties(Titan PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#
#target_compile_definitions(Titan PUBLIC GRAPHICS)
#target_compile_definitions(Titan PUBLIC CONSTRAINTS) # defines the CONSTRAINTS preprocessor variable (enables local constraints)
#
#if (NOT TITAN_SHARED_BUILD)
#    target_compile_definitions(Titan PUBLIC _MWAITXINTRIN_H_INCLUDED)
#    target_compile_features(Titan PUBLIC cxx_std_11) # same as above, may be unnecessary
#endif()
#
#find_package(CUDA REQUIRED) # find and include CUDA
#if (CUDA_FOUND)
#    message(STATUS "CUDA FOUND")
#    target_include_directories(Titan PUBLIC ${CUDA_INCLUDE_DIRS})
#    target_link_libraries(Titan PRIVATE cuda) # cudart
#else()
#    message(STATUS "CUDA NOT FOUND")
#endif()
#
#if (UNIX)
#    find_package(OpenGL REQUIRED) # find and include OpenGL
#    if (OpenGL_FOUND)
#        message(STATUS "OPENGL FOUND")
#        target_include_directories(Titan PRIVATE ${OpenGL_INCLUDE_DIRS})
#        target_link_libraries(Titan PRIVATE ${OpenGL_LIBRARIES} GL)
#    endif()
#else()
#    find_package(OPENGL REQUIRED) # find and include OpenGL
#    if (OPENGL_FOUND)
#        message(STATUS "OPENGL FOUND")
#        target_include_directories(Titan PRIVATE ${OPENGL_INCLUDE_DIRS})
#        target_link_libraries(Titan PRIVATE ${OPENGL_LIBRARIES})
#    endif()
#endif()
#
#find_package(glfw3 CONFIG REQUIRED)
#if (glfw3_FOUND)
#    message(STATUS "GLFW FOUND")
#    target_include_directories(Titan PRIVATE ${glfw3_INCLUDE_DIRS})
#    target_link_libraries(Titan PRIVATE ${glfw3_LIBRARIES})
#endif()
#
#find_package(GLEW REQUIRED) # GLEW
#if (GLEW_FOUND)
#    message(STATUS "GLEW FOUND")
#    target_include_directories(Titan PRIVATE ${GLEW_INCLUDE_DIRS})
#    target_link_libraries(Titan PRIVATE ${GLEW_LIBRARIES} glfw)
#endif()
#
#find_package(glm CONFIG REQUIRED) # glm
#if (glm_FOUND)
#    message(STATUS "GLM FOUND")
#    target_include_directories(Titan PRIVATE ${glm_INCLUDE_DIRS})
#    target_link_libraries(Titan PRIVATE ${glm_LIBRARIES} glm)
#endif()
#
#find_package(pybind11 REQUIRED)  #pybind11
#if (pybind11_FOUND)
#    message(STATUS "pybind11 FOUND")
#    #no need ot target_include or target_link. done later through add module
#else()
#    message(STATUS "pybind11 NOT FOUND")
#endif()
#
#
#target_include_directories(Titan PUBLIC
#        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Titan>
#        $<INSTALL_INTERFACE:include>)
#
#set_target_properties(Titan PROPERTIES PUBLIC_HEADER "${TITAN_HEADER_FILES}")
#
#install(TARGETS Titan
#        EXPORT TitanTargets
#        PUBLIC_HEADER DESTINATION include/Titan
#        LIBRARY DESTINATION bin
#        ARCHIVE DESTINATION lib
#        RUNTIME DESTINATION bin
#        )
#
#if (TITAN_SHARED_BUILD)
#    install(EXPORT TitanTargets
#            FILE TitanConfig.cmake
#            DESTINATION share/titan)
#else()
#    install(EXPORT TitanTargets
#            FILE TitanTargets.cmake
#            DESTINATION share/titan)
#
#    configure_file(cmake/TitanConfig.cmake.in2
#            ${CMAKE_INSTALL_PREFIX}/share/titan/TitanConfig.cmake @ONLY)
#endif()
#
#
##Create Library with .cu files
#add_library(cu_lib ${LOCH_FILES})
#target_include_directories(cu_lib PRIVATE ${LOCH_INCLUDE_DIR}} ${CUDA_INCLUDE_DIRS} )
#set_target_properties(cu_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(cu_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(cu_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#
##target_compile_definitions(cu_lib PRIVATE CONSTRAINTS) # defines the CONSTRAINTS preprocessor variable (enables local constraints)
##target_compile_definitions(cu_lib PRIVATE GRAPHICS) # defines the GRAPHICS preprocessor variable
#
#target_compile_features(cu_lib PUBLIC cxx_std_11)
#target_link_libraries(cu_lib PRIVATE cuda cudart glm)
#
#if ( WIN32 ) # use GLFW on Windows
#    target_link_libraries(cu_lib PRIVATE GLEW::GLEW)
#    target_link_libraries(cu_lib PRIVATE glfw)
#endif()
#
##PYBIND11
##CREATE PYTHON BINDINGS
#pybind11_add_module(bindloch ${LOCH_BIND_FILES} ${LOCH_HEADER_FILES})
#
##Add linking dependencies to the binding library
#target_include_directories(bindloch PRIVATE ${LOCH_BIND_INCLUDE_DIR}
#        ${LOCH_INCLUDE_DIR})
#add_dependencies(bindloch cu_lib)
#target_link_libraries(bindloch PRIVATE cu_lib)
#

#
#cmake_minimum_required(VERSION 3.10)
#project(Loch LANGUAGES C CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)
#
#set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++11
#
#set(LOCH_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
#set(LOCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/Loch)
#
#set(LOCH_SOURCE_FILES ${LOCH_SOURCE_DIR}/vec.cu ${LOCH_SOURCE_DIR}/sim.cu ${LOCH_SOURCE_DIR}/shader.cpp
#        ${LOCH_SOURCE_DIR}/mass.cu ${LOCH_SOURCE_DIR}/spring.cu
#        ${LOCH_SOURCE_DIR}/object.cu ${LOCH_SOURCE_DIR}/graphics.cpp)
#
#set(LOCH_HEADER_FILES ${LOCH_INCLUDE_DIR}/sim.h ${LOCH_INCLUDE_DIR}/mass.h ${LOCH_INCLUDE_DIR}/object.h
#        ${LOCH_INCLUDE_DIR}/spring.h ${LOCH_INCLUDE_DIR}/vec.h ${LOCH_INCLUDE_DIR}/graphics.h ${LOCH_INCLUDE_DIR}/shader.h
#        ${LOCH_INCLUDE_DIR}/stlparser.h)
#
#add_executable(Loch main.cpp ${LOCH_SOURCE_FILES} ${LOCH_HEADER_FILES})
#
#set_target_properties(Loch PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # needed for library compilation
#set_target_properties(Loch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#
#target_compile_definitions(Loch PUBLIC GRAPHICS)
#target_compile_definitions(Loch PUBLIC CONSTRAINTS) # defines the CONSTRAINTS preprocessor variable (enables local constraints)
#
#find_package(CUDA REQUIRED) # find and include CUDA
#if (CUDA_FOUND)
#    message(STATUS "CUDA FOUND")
#    target_include_directories(Loch PUBLIC ${CUDA_INCLUDE_DIRS})
#    target_link_libraries(Loch PRIVATE cuda) # cudart
#else()
#    message(STATUS "CUDA NOT FOUND")
#endif()
#
#if (UNIX)
#    find_package(OpenGL REQUIRED) # find and include OpenGL
#    if (OpenGL_FOUND)
#        message(STATUS "OPENGL FOUND")
#        target_include_directories(Loch PRIVATE ${OpenGL_INCLUDE_DIRS})
#        target_link_libraries(Loch PRIVATE ${OpenGL_LIBRARIES} GL)
#    endif()
#else()
#    find_package(OPENGL REQUIRED) # find and include OpenGL
#    if (OPENGL_FOUND)
#        message(STATUS "OPENGL FOUND")
#        target_include_directories(Loch PRIVATE ${OPENGL_INCLUDE_DIRS})
#        target_link_libraries(Loch PRIVATE ${OPENGL_LIBRARIES})
#    endif()
#endif()
#
#find_package(glfw3 CONFIG REQUIRED)
#if (glfw3_FOUND)
#    message(STATUS "GLFW FOUND")
#    target_include_directories(Loch PRIVATE ${glfw3_INCLUDE_DIRS})
#    target_link_libraries(Loch PRIVATE ${glfw3_LIBRARIES})
#endif()
#
#find_package(GLEW REQUIRED) # GLEW
#if (GLEW_FOUND)
#    message(STATUS "GLEW FOUND")
#    target_include_directories(Loch PRIVATE ${GLEW_INCLUDE_DIRS})
#    target_link_libraries(Loch PRIVATE ${GLEW_LIBRARIES} glfw)
#endif()
#
#find_package(glm CONFIG REQUIRED) # glm
#if (glm_FOUND)
#    message(STATUS "GLM FOUND")
#    target_include_directories(Loch PRIVATE ${glm_INCLUDE_DIRS})
#    target_link_libraries(Loch PRIVATE ${glm_LIBRARIES} glm)
#endif()


#set(PYTHON_EXECUTABLE C:/Users/rcorr/Anaconda3/python.exe)
cmake_minimum_required(VERSION 3.11)
project(Loch LANGUAGES CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++11
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src) # set SOURCE_DIR to src directory
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/Titan) # set SOURCE_DIR to src directory

include_directories(include/Titan) # include the include directory (can find headers there)

#SET CPP PLOCH PROJECT
set(LOCH_SOURCE_FILES  src/object.cu src/graphics.cpp src/mass.cu src/sim.cu src/spring.cu src/vec.cu src/shader.cpp)
set(LOCH_HEADER_FILES ${INCLUDE_DIR}/object.h ${INCLUDE_DIR}/graphics.h ${INCLUDE_DIR}/mass.h ${INCLUDE_DIR}/sim.h ${INCLUDE_DIR}/spring.h ${INCLUDE_DIR}/vec.h)
set(LOCH_FILES ${LOCH_HEADER_FILES} ${LOCH_SOURCE_FILES})
#SET PYTHON BINDING DIRECTORIES
set(LOCH_BIND_SOURCE_FILES  bind/src/pysimulation.cpp bind/src/pymass.cpp bind/src/pyspring.cpp bind/src/pyobject.cpp bind/src/bind.cpp
        bind/src/bind_mass.cpp  bind/src/bind_sim.cpp bind/src/bind_spring.cpp bind/src/bind_object.cpp )
set(LOCH_BIND_HEADER_FILES bind/include/pysimulation.h bind/include/pyobject.h bind/include/pyspring.h bind/include/pymass.h)
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

#Create Library with .cu files
add_library(cu_lib ${LOCH_FILES})
target_include_directories(cu_lib PRIVATE ${INCLUDE_DIR}} ${CUDA_INCLUDE_DIRS} )
set_target_properties(cu_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cu_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cu_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

#target_compile_definitions(cu_lib PRIVATE CONSTRAINTS) # defines the CONSTRAINTS preprocessor variable (enables local constraints)
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

set_target_properties( bindloch
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python_module"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python_module"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python_module"
        )

