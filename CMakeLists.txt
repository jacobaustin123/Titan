cmake_minimum_required(VERSION 3.10)
project("Titan" 
        VERSION 0.8
        DESCRIPTION "CUDA physics simulator for soft robotics and reinforcement learning"
        LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++11

option(TITAN_BUILD_TESTS "Enable unit tests to be built." OFF)
option(TITAN_SHARED_BUILD "Enables shared library build (static by default)" OFF)
option(GRAPHICS "Enables Titan OpenGL graphics support" ON)
option(CONSTRAINTS "Enables Titan support for constraints" ON)
option(VERLET "Makes Titan use a Verlet integration scheme instead of a naive Euler scheme" OFF)
option(RK2 "Makes Titan use an RK2 integration scheme instead of a naive Euler scheme. Requires C++17." OFF)
option(TITAN_INSTALL "Install Titan files at path specified by CMAKE_INSTALL_PREFIX" OFF)

if (TITAN_BUILD_TESTS AND GRAPHICS)
    message(WARNING "Cannot run tests while building with graphics support. Disabling graphics support.")
    set(GRAPHICS OFF)
endif()

# Automate the choosing of config
if (NOT CMAKE_BUILD_TYPE)
    # if binary directory ends with "release", use release mode
    if (${PROJECT_BINARY_DIR} MATCHES "release$")
        set(CMAKE_BUILD_TYPE RELEASE)
    # otherwise, use debug mode
    else()
        set(CMAKE_BUILD_TYPE DEBUG)
    endif()
endif()
message(STATUS "Compiling in ${CMAKE_BUILD_TYPE} mode")

# fix GLM compiler warning (try -Xcudafe --display_error_number)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe=\"--diag_suppress=2977\"")

set(TITAN_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(TITAN_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/Titan)

# include all source (*.cpp/*.cu) files
set(TITAN_SOURCE_FILES 
    ${TITAN_SOURCE_DIR}/vec.cu
    ${TITAN_SOURCE_DIR}/sim.cu
    ${TITAN_SOURCE_DIR}/shader.cpp
    ${TITAN_SOURCE_DIR}/mass.cu
    ${TITAN_SOURCE_DIR}/spring.cu
    ${TITAN_SOURCE_DIR}/object.cu
    ${TITAN_SOURCE_DIR}/graphics.cpp
)

# include all header files (*.h)
set(TITAN_HEADER_FILES 
    ${TITAN_INCLUDE_DIR}/sim.h
    ${TITAN_INCLUDE_DIR}/mass.h
    ${TITAN_INCLUDE_DIR}/object.h
    ${TITAN_INCLUDE_DIR}/spring.h
    ${TITAN_INCLUDE_DIR}/vec.h
    ${TITAN_INCLUDE_DIR}/graphics.h
    ${TITAN_INCLUDE_DIR}/shader.h
    ${TITAN_INCLUDE_DIR}/stlparser.h
)

# if building as a shared library
if (TITAN_SHARED_BUILD)
    message(STATUS "Building Titan as a shared library...")
    set(BUILD_SHARED_LIBS ON)
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(POSITION_INDEPENDENT_CODE ON)

add_library(Titan ${TITAN_SOURCE_FILES} ${TITAN_HEADER_FILES})

# if enabled, compiles with support for graphics using OpenGL.
if (GRAPHICS)
    message(STATUS "Enabling support for OpenGL graphics...")
    target_compile_definitions(Titan PUBLIC GRAPHICS)
endif()

# if enabled, compiles with support for constraints.
if (CONSTRAINTS)
    message(STATUS "Enabling support for Constraints..")
    target_compile_definitions(Titan PUBLIC CONSTRAINTS) 
endif()

# if enabled, uses a Verlet integration scheme
if (VERLET)
    message(STATUS "Enabling support for Verlet integration..")
    target_compile_definitions(Titan PUBLIC VERLET) 
endif()

# if enabled, uses an RK2 integration scheme
if (RK2)
    message(STATUS "Enabling support for RK2 integration (overrides Verlet and other methods)...")
    target_compile_features(Titan PRIVATE cxx_std_17)
    target_compile_definitions(Titan PUBLIC RK2) 
endif()

# needed for library compilation (see https://stackoverflow.com/a/51353324)
set_target_properties(Titan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Titan PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# these fix an obscure bug while building a shared library
if (NOT TITAN_SHARED_BUILD)
    target_compile_definitions(Titan PUBLIC _MWAITXINTRIN_H_INCLUDED)
    target_compile_features(Titan PUBLIC cxx_std_11)
endif()

set(Titan_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/include)
set(Titan_LIBRARIES Titan)

find_package(CUDA REQUIRED) # find and include CUDA
if (CUDA_FOUND)
    message(STATUS "CUDA FOUND")
    target_include_directories(Titan PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(Titan PRIVATE cuda) # cudart
    set(Titan_INCLUDE_DIRS ${Titan_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
    set(Titan_LIBRARIES ${Titan_LIBRARIES} ${CUDA_LIBRARIES})
else()
    message(STATUS "CUDA NOT FOUND")
endif()

# find and include OpenGL for graphics support
set(OpenGL_GL_PREFERENCE LEGACY)
if (UNIX)
    find_package(OpenGL REQUIRED)
    if (OpenGL_FOUND)
        message(STATUS "OPENGL FOUND")
        target_include_directories(Titan PRIVATE ${OpenGL_INCLUDE_DIRS})
        target_link_libraries(Titan PRIVATE ${OpenGL_LIBRARIES} GL)
        set(Titan_INCLUDE_DIRS ${Titan_INCLUDE_DIRS} ${OpenGL_INCLUDE_DIRS})
        set(Titan_LIBRARIES ${Titan_LIBRARIES} ${OpenGL_LIBRARIES})
    endif()
else()
    find_package(OPENGL REQUIRED) # find and include OpenGL
    if (OPENGL_FOUND)
        message(STATUS "OPENGL FOUND")
        target_include_directories(Titan PRIVATE ${OPENGL_INCLUDE_DIRS})
        target_link_libraries(Titan PRIVATE ${OPENGL_LIBRARIES})
        set(Titan_INCLUDE_DIRS ${Titan_INCLUDE_DIRS} ${OpenGL_INCLUDE_DIRS})
        set(Titan_LIBRARIES ${Titan_LIBRARIES} ${OpenGL_LIBRARIES})
    endif()
endif()

# find and include glfw for graphics support 
find_package(glfw3 CONFIG REQUIRED)
if (glfw3_FOUND)
    message(STATUS "GLFW FOUND")
    target_include_directories(Titan PRIVATE ${glfw3_INCLUDE_DIRS})
    target_link_libraries(Titan PRIVATE ${glfw3_LIBRARIES} glfw)
    set(Titan_INCLUDE_DIRS ${Titan_INCLUDE_DIRS} ${glfw3_INCLUDE_DIRS})
    set(Titan_LIBRARIES ${Titan_LIBRARIES} ${glfw3_LIBRARIES})
endif()

# find and include GLEW for graphics support
find_package(GLEW REQUIRED) # GLEW
if (GLEW_FOUND)
    message(STATUS "GLEW FOUND")
    target_include_directories(Titan PRIVATE ${GLEW_INCLUDE_DIRS})
    target_link_libraries(Titan PRIVATE ${GLEW_LIBRARIES})
    set(Titan_INCLUDE_DIRS ${Titan_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
    set(Titan_LIBRARIES ${Titan_LIBRARIES} ${GLEW_LIBRARIES})
endif()

# find and include glm for graphics support
find_package(glm CONFIG REQUIRED) # glm
if (glm_FOUND)
    message(STATUS "GLM FOUND")
    target_include_directories(Titan PRIVATE ${glm_INCLUDE_DIRS})
    target_link_libraries(Titan PRIVATE ${glm_LIBRARIES} glm)
    set(Titan_INCLUDE_DIRS ${Titan_INCLUDE_DIRS} ${glm_INCLUDE_DIRS})
    set(Titan_LIBRARIES ${Titan_LIBRARIES} ${glm_LIBRARIES})
endif()

# message(${Titan_LIBRARIES})
# message(${Titan_INCLUDE_DIRS})

target_include_directories(Titan PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Titan>
        $<INSTALL_INTERFACE:include>)

set_target_properties(Titan PROPERTIES PUBLIC_HEADER "${TITAN_HEADER_FILES}")

if (TITAN_INSTALL)
    install(TARGETS Titan
            EXPORT TitanTargets
            PUBLIC_HEADER DESTINATION include/Titan
            LIBRARY DESTINATION bin
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin
            )

    if (TITAN_SHARED_BUILD)
        install(EXPORT TitanTargets
                FILE TitanConfig.cmake
                DESTINATION share/titan)
    else()
        install(EXPORT TitanTargets
                FILE TitanTargets.cmake
                DESTINATION share/titan)

        configure_file(cmake/TitanConfig.cmake.in2
                ${CMAKE_INSTALL_PREFIX}/share/titan/TitanConfig.cmake @ONLY)
    endif()
endif()

set(GTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/googletest/googletest)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib ${PROJECT_BINARY_DIR}/lib)

if (TITAN_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(${PROJECT_SOURCE_DIR}/test ${PROJECT_BINARY_DIR}/test)
endif()
