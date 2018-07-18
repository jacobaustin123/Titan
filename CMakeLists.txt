set(PYTHON_EXECUTABLE C:/Users/rcorr/Anaconda3/python.exe)
cmake_minimum_required(VERSION 3.9)
project(Loch LANGUAGES CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)

set(BUILD_SHARED_LIBS OFF)
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++11
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14") # same thing, may be unnecessary
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src") # set SOURCE_DIR to src directory
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include") # set SOURCE_DIR to src directory

include_directories(include) # include the include directory (can find headers there)

set(LOCH_SOURCE_FILES src/graphics.cpp src/main.cpp src/mass.cu src/object.cu src/sim.cu src/spring.cu src/vec.cu)
set(LOCH_HEADER_FILES include/graphics.h include/mass.h include/object.h include/sim.h include/spring.h include/vec.h)
set(LOCH_BIND_FILES bind/bind.cpp bind/bind_mass.cpp bind/bind_object.cpp  bind/bind_sim.cpp bind/bind_spring.cpp)

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
add_library(cu_lib2 ${LOCH_SOURCE_FILES} ${LOCH_HEADER_FILES})
target_include_directories(cu_lib2 PRIVATE ${INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
set_target_properties(cu_lib2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cu_lib2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cu_lib2 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_features(cu_lib2 PUBLIC cxx_std_11)
target_link_libraries(cu_lib2 PRIVATE cuda)
target_link_libraries(cu_lib2 PRIVATE cudart)

#Create binding
pybind11_add_module(bindloch ${LOCH_BIND_FILES} ${LOCH_HEADER_FILES})

#Add linking dependedncies to the binding library
target_include_directories(bindloch PRIVATE include)
add_dependencies(bindloch cu_lib2)
target_link_libraries(bindloch PRIVATE cu_lib2)



#add_executable(nographics src/main.cpp ${SOURCE_FILES} ${HEADERS}) # create nographics target
#target_compile_features(nographics PUBLIC cxx_std_11)
#set_target_properties(nographics PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # allows declarations and implementations to be separated
#target_link_libraries(nographics PRIVATE cuda)
#
#add_executable(graphics src/main.cpp ${SOURCE_FILES} ${HEADERS}) # create graphics target
#target_compile_definitions(graphics PRIVATE GRAPHICS) # defines the GRAPHICS preprocessor variable
#
#target_compile_features(graphics PUBLIC cxx_std_11) # same as above
#set_target_properties(graphics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(graphics PRIVATE cuda)
#
#target_link_libraries(graphics PRIVATE glm)
#
#if ( WIN32 ) # use GLFW on Windows
#    target_link_libraries(graphics PRIVATE GLEW::GLEW)
#    target_link_libraries(graphics PRIVATE glfw)
#else() # use SDL2 on Mac
#    target_link_libraries(graphics PRIVATE GLEW)
#    target_link_libraries(graphics PRIVATE gl)
#    target_link_libraries(graphics PRIVATE sdl2)
#    target_compile_definitions(graphics PRIVATE SDL2)
#endif()