# Configure test coverage
######################################################
# Physics tests
######################################################

add_executable(physics_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/physics/simple_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physics/energy_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physics/multiagent_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physics/rotate_unittest.cpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(physics_unittest PRIVATE -g -Wall)
else()
	target_compile_options(physics_unittest PRIVATE -g -Wall -Werror -Wextra)
endif()

target_include_directories(physics_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Titan_INCLUDE_DIRS}
)

target_link_libraries(physics_unittest titan_gtest_main ${Titan_LIBRARIES})
add_test(physics_unittest physics_unittest)