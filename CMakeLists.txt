cmake_minimum_required(VERSION 3.7)
project(Loch)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_PREFIX "C:/Users/CreativeMachines/Desktop")

set(LOCH_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(LOCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/Loch)

message(STATUS ${LOCH_SOURCE_DIR} " " ${LOCH_INCLUDE_DIR})

set(LOCH_SOURCE_FILES ${LOCH_SOURCE_DIR}/vec.cpp ${LOCH_SOURCE_DIR}/sim.cpp
        ${LOCH_SOURCE_DIR}/sim.cpp ${LOCH_SOURCE_DIR}/mass.cpp ${LOCH_SOURCE_DIR}/spring.cpp
        ${LOCH_SOURCE_DIR}/object.cpp ${LOCH_SOURCE_DIR}/graphics.cpp)

set(LOCH_HEADER_FILES ${LOCH_INCLUDE_DIR}/sim.h ${LOCH_INCLUDE_DIR}/mass.h ${LOCH_INCLUDE_DIR}/object.h
        ${LOCH_INCLUDE_DIR}/spring.h ${LOCH_INCLUDE_DIR}/vec.h ${LOCH_INCLUDE_DIR}/graphics.h)

if (LOCH_SHARED_BUILD)
    set(BUILD_SHARED_LIBS ON)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(Loch ${LOCH_SOURCE_FILES} ${LOCH_HEADER_FILES})

target_compile_definitions(Loch PRIVATE GRAPHICS)

find_package(OPENGL REQUIRED)
if (OPENGL_FOUND)
    message(STATUS "OPENGL FOUND")
    target_include_directories(Loch PRIVATE ${OPENGL_INCLUDE_DIRS})
    target_link_libraries(Loch PRIVATE ${OPENGL_LIBRARIES})
endif()

find_package(glfw3 REQUIRED)
if (glfw3_FOUND)
    message(STATUS "GLFW FOUND")
    target_include_directories(Loch PRIVATE ${glfw3_INCLUDE_DIRS})
    target_link_libraries(Loch PRIVATE ${glfw3_LIBRARIES} glfw)
endif()

find_package(GLEW REQUIRED)
if (GLEW_FOUND)
    message(STATUS "GLEW FOUND")
    target_include_directories(Loch PRIVATE ${GLEW_INCLUDE_DIRS})
    target_link_libraries(Loch PRIVATE ${GLEW_LIBRARIES})
endif()

find_package(glm CONFIG REQUIRED)
if (glm_FOUND)
    message(STATUS "GLM FOUND")
    target_include_directories(Loch PRIVATE ${glm_INCLUDE_DIRS})
    target_link_libraries(Loch PRIVATE ${glm_LIBRARIES})
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

install(EXPORT LochTargets
        FILE LochTargets.cmake
        DESTINATION share/Loch)

#install(DIRECTORY ${LOCH_INCLUDE_DIR} DESTINATION include/Loch)
#install(TARGETS Loch DESTINATION lib)
#install(EXPORT Loch FILE LochConfig.cmake DESTINATION

#set(Loch_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include/Loch)
#set(Loch_LIBRARIES ${CMAKE_INSTALL_PREFIX}/lib/Loch)

configure_file(cmake/LochConfig.cmake.in2
        ${CMAKE_INSTALL_PREFIX}/share/Loch/LochConfig.cmake @ONLY)

#add_subdirectory(tests/test1)
#add_subdirectory(tests/test2)