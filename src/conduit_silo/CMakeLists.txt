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
# Add the conduit-silo library
################################

#
# Specify all headers
#            
set(conduit_silo_headers conduit_silo.h Conduit_Silo_Exports.h)

#
# Specify all sources
#
set(conduit_silo_sources conduit_silo.cpp)

#
# Note: headers are added here so they show up in project file generators
# (such as xcode, eclipse, etc)
#
add_library(conduit_silo SHARED ${conduit_silo_sources} ${conduit_silo_headers})

#
# Link with conduit and silo
#
target_link_libraries(conduit_silo conduit ${SILO_LIBRARIES})

##################################
# Install Targets for conduit lib
##################################

install(FILES ${conduit_silo_headers} DESTINATION include)
install(TARGETS conduit_silo
  EXPORT conduit-silo-lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

