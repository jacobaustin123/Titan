cmake_minimum_required(VERSION 3.9)
project(Loch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")


set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(include)

set(SOURCE_FILES src/vec.cu src/sim.cu src/sim.cu src/mass.cu src/spring.cu src/object.cu src/graphics.cpp)

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA FOUND")
    include_directories(${CUDA_INCLUDE_DIRS})
    link_libraries(${CUDA_LIBRARIES})
else()
	message(STATUS "CUDA NOT FOUND")
endif()

file(GLOB HEADERS "*.h")

add_executable(main src/main.cpp ${SOURCE_FILES} ${HEADERS})
target_compile_features(main PUBLIC cxx_std_11)

set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main cuda)

install(TARGETS main DESTINATION ${CMAKE_BINARY_DIR})
