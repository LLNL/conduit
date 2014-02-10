################################
# Add our library
################################


configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/Conduit_Config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/Conduit_Config.h")
#
# Collect all headers
#            
file(GLOB conduit_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
list(APPEND conduit_headers ${CMAKE_CURRENT_BINARY_DIR}/Conduit_Config.h)

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
    )

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

