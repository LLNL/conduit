#############################################################################
# Copyright (c) 2014, Lawrence Livermore National Security, LLC
# Produced at the Lawrence Livermore National Laboratory. 
# 
# All rights reserved.
# 
# This source code cannot be distributed without further review from 
# Lawrence Livermore National Laboratory.
#############################################################################

################################
# Standard CMake Options
################################


# Fail if someone tries to config an in-source build.
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
   message(FATAL_ERROR "In-source builds are not supported. Please remove CMakeCache.txt from the 'src' dir and configure an out-of-source build in another directory.")
endif()

# enable creation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


macro(ENABLE_WARNINGS)
    # set the warning levels we want to abide by
    if(CMAKE_BUILD_TOOL MATCHES "(msdev|devenv|nmake)")
        add_definitions(/W2)
    else()
        add_definitions(-Wall -Wextra)
    endif()
endmacro()
