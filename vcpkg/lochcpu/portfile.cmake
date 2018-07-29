# Common Ambient Variables:
#   CURRENT_BUILDTREES_DIR    = ${VCPKG_ROOT_DIR}\buildtrees\${PORT}
#   CURRENT_PACKAGES_DIR      = ${VCPKG_ROOT_DIR}\packages\${PORT}_${TARGET_TRIPLET}
#   CURRENT_PORT_DIR          = ${VCPKG_ROOT_DIR}\ports\${PORT}
#   PORT                      = current port name (zlib, etc)
#   TARGET_TRIPLET            = current triplet (x86-windows, x64-windows-static, etc)
#   VCPKG_CRT_LINKAGE         = C runtime linkage type (static, dynamic)
#   VCPKG_LIBRARY_LINKAGE     = target library linkage type (static, dynamic)
#   VCPKG_ROOT_DIR            = <C:\path\to\current\vcpkg>
#   VCPKG_TARGET_ARCHITECTURE = target architecture (x64, x86, arm)

include(vcpkg_common_functions)

#set(SOURCE_PATH ${CURRENT_BUILDTREES_DIR}/src/zlib1211-2)
#vcpkg_download_distfile(ARCHIVE
#    URLS "http://zlib.net/zlib1211.zip"
#    FILENAME "zlib1211-2.zip"
#    SHA512 9069fe4a9bfae1d45bcf000404c9f89605741b67987b8ca412b84ba937ed1b7bba4b8b174b6f9bc6814776def5f4b34ec7785cd84aa410002d87ddd1b507f11a
#)

vcpkg_extract_source_archive(${VCPKG_ROOT_DIR}/downloads/Loch.zip)

#vcpkg_from_github(OUT_SOURCE_PATH SOURCE_PATH
#        REPO ja3067/LochCPU
#        REF 70a6c2217f1b5dd199123e419e51853fe7a290cc
#        SHA512 1
#        HEAD_REF package
#        )

#vcpkg_from_github(OUT_SOURCE_PATH SOURCE_PATH
#        REPO Microsoft/vcpkg
#        REF 2035d0124dd01ea3ebabfdcc99c7275f723cab13
#        SHA512 8291639d16ebf9f5394f601b7951028dbb095dfece47c57deb1b2e43f5a893c0b128632d82157bfe1f0ce26f21837a598a638cc399e16d1906d825a3ed5e10ac
#        HEAD_REF master
#        )

set(SOURCE_PATH ${CURRENT_BUILDTREES_DIR}/src/Loch)

message(STATUS ${VCPKG_LIBRARY_LINKAGE})

if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
    message(STATUS "BUILDING AS STATIC LIBRARY")
#    if (${TARGET_TRIPLET} STREQUAL "x64-windows-static")
#        message(FATAL_ERROR "Loch currently does not support static linkage on Windows. Please use the x64-windows triplet!")
#    else()
        vcpkg_configure_cmake(
                SOURCE_PATH ${SOURCE_PATH}
                PREFER_NINJA
                OPTIONS
                -DLOCH_SHARED_BUILD=OFF
        )
#    endif()
else()
    message(STATUS "BUILDING AS SHARED LIBRARY")
    vcpkg_configure_cmake(
            SOURCE_PATH ${SOURCE_PATH}
            PREFER_NINJA
            OPTIONS
            -DLOCH_SHARED_BUILD=ON
    )
endif()


vcpkg_install_cmake()
file(
        REMOVE_RECURSE
        ${CURRENT_PACKAGES_DIR}/debug/include
        ${CURRENT_PACKAGES_DIR}/debug/share
)

# Handle copyright
file(INSTALL ${SOURCE_PATH}/vcpkg/copyright.txt DESTINATION ${CURRENT_PACKAGES_DIR}/share/lochcpu RENAME copyright)