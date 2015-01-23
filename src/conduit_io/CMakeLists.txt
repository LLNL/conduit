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
# Add the conduit-io library
################################

ENABLE_WARNINGS()

if(SILO_FOUND)
  SET(CONDUIT_IO_ENABLE_SILO TRUE)
endif()

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/Conduit_IO_Config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/Conduit_IO_Config.h")

#
# Specify all headers
#            
set(conduit_io_headers 
    conduit_io.h 
    Conduit_IO_Exports.h
    ${CMAKE_CURRENT_BINARY_DIR}/Conduit_IO_Config.h)

#
# Specify all sources
#
set(conduit_io_sources conduit_io.cpp)

if(SILO_FOUND)
    list(APPEND conduit_io_headers conduit_silo.h)
    list(APPEND conduit_io_sources conduit_silo.cpp)
endif()


#
# Note: headers are added here so they show up in project file generators
# (such as xcode, eclipse, etc)
#
add_library(conduit_io SHARED ${conduit_io_sources} ${conduit_io_headers})

#
# Link with conduit and silo (if enabled)
#
target_link_libraries(conduit_io conduit)

if(SILO_FOUND)
    target_link_libraries(conduit_io conduit ${SILO_LIBRARIES})
endif()

##################################
# Install Targets for conduit lib
##################################

install(FILES ${conduit_io_headers} DESTINATION include)
install(TARGETS conduit_io
  EXPORT conduit-io-lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

