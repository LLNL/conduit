# Find rapidjson headers and lib.
# This module defines RAPIDJSON_INCLUDE_DIR, directory containing headers

set(RAPIDJSON_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/thirdparty_builtin/rapidjson/include)

if (RAPIDJSON_INCLUDE_DIR)
  set(RAPIDJSON_FOUND TRUE)
else ()
  set(RAPIDJSON_FOUND FALSE)
endif()

if (RAPIDJSON_FOUND)
  if (NOT RAPIDJSON_FIND_QUIETLY)
    message(STATUS "RapidJSON headers found in: ${RAPIDJSON_INCLUDE_DIR}")
  endif ()
else ()
  message(STATUS "RapidJSON headers NOT found.")
endif ()

mark_as_advanced(RAPIDJSON_INCLUDE_DIR)