################################
# Add our library
################################


configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/Conduit_Config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/Conduit_Config.h")
#
# Specify all headers
#            
set(conduit_headers
    Bitwidth_Style_Types.h
    DataArray.h
    Endianness.h
    Node.h
    conduit.h
    Core.h
    DataType.h
    Error.h
    Schema.h
    Utils.h
    ${CMAKE_CURRENT_BINARY_DIR}/Conduit_Config.h
    )

#
# Specify all sources
#
set(conduit_sources
    Core.cpp
    Error.cpp
    Endianness.cpp 
    DataType.cpp
    DataArray.cpp
    Node.cpp
    Schema.cpp
    Utils.cpp
    )

#
# Note: headers are added here so they show up in project file generators
# (such as xcode, eclipse, etc)
#
add_library(conduit SHARED ${conduit_sources} ${conduit_headers})

##################################
# Install Targets for conduit lib
##################################

install(FILES ${conduit_headers} DESTINATION include)
install(TARGETS conduit
  EXPORT conduit-lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

