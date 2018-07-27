#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Loch" for configuration "Release"
set_property(TARGET Loch APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Loch PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/Loch.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/Loch.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS Loch )
list(APPEND _IMPORT_CHECK_FILES_FOR_Loch "${_IMPORT_PREFIX}/lib/Loch.lib" "${_IMPORT_PREFIX}/bin/Loch.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
