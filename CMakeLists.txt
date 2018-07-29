cmake_minimum_required(VERSION 3.11)
project(LochCPU)

set(CMAKE_CXX_STANDARD 17)

set(LOCH_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(LOCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/LochCPU)

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

add_library(LochCPU ${LOCH_SOURCE_FILES} ${LOCH_HEADER_FILES})

target_compile_definitions(LochCPU PRIVATE GRAPHICS)

find_package(OPENGL REQUIRED)
if (OPENGL_FOUND)
    message(STATUS "OPENGL FOUND")
    target_include_directories(LochCPU PRIVATE ${OPENGL_INCLUDE_DIRS})
    target_link_libraries(LochCPU PRIVATE ${OPENGL_LIBRARIES})
endif()

find_package(glfw3 REQUIRED)
if (glfw3_FOUND)
    message(STATUS "GLFW FOUND")
    target_include_directories(LochCPU PRIVATE ${glfw3_INCLUDE_DIRS})
    target_link_libraries(LochCPU PRIVATE ${glfw3_LIBRARIES} glfw)
endif()

find_package(GLEW REQUIRED)
if (GLEW_FOUND)
    message(STATUS "GLEW FOUND")
    target_include_directories(LochCPU PRIVATE ${GLEW_INCLUDE_DIRS})
    target_link_libraries(LochCPU PRIVATE ${GLEW_LIBRARIES})
endif()

find_package(glm CONFIG REQUIRED)
if (glm_FOUND)
    message(STATUS "GLM FOUND")
    target_include_directories(LochCPU PRIVATE ${glm_INCLUDE_DIRS})
    target_link_libraries(LochCPU PRIVATE ${glm_LIBRARIES})
endif()

target_include_directories(LochCPU PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/LochCPU>
        $<INSTALL_INTERFACE:include>)

set_target_properties(LochCPU PROPERTIES PUBLIC_HEADER "${LOCH_HEADER_FILES}")

install(TARGETS LochCPU
        EXPORT LochCPUTargets
        PUBLIC_HEADER DESTINATION include/LochCPU
        LIBRARY DESTINATION bin
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

if (LOCH_SHARED_BUILD)
    message(STATUS "EXPORTING AS SHARED")

    install(EXPORT LochCPUTargets
            FILE LochCPUConfig.cmake
            DESTINATION share/lochcpu)
else()
    message(STATUS "EXPORTING AS STATIC")

    install(EXPORT LochCPUTargets
            FILE LochCPUTargets.cmake
            DESTINATION share/lochcpu)

    configure_file(cmake/LochConfig.cmake.in2
            ${CMAKE_INSTALL_PREFIX}/share/lochcpu/LochCPUConfig.cmake @ONLY)
endif()

#install(DIRECTORY ${LOCH_INCLUDE_DIR} DESTINATION include/LochCPU)
#install(TARGETS LochCPU DESTINATION lib)
#install(EXPORT LochCPU FILE LochConfig.cmake DESTINATION

#set(Loch_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include/LochCPU)
#set(Loch_LIBRARIES ${CMAKE_INSTALL_PREFIX}/lib/LochCPU)

#configure_file(cmake/LochConfig.cmake.in2
#        ${CMAKE_INSTALL_PREFIX}/share/LochCPU/LochConfig.cmake @ONLY)

#add_subdirectory(tests/test1)
#add_subdirectory(tests/test2)