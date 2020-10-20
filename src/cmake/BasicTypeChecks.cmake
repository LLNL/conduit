# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

################################
# Checks for type sizes
################################

INCLUDE (CheckTypeSize)

CHECK_TYPE_SIZE("char"          CONDUIT_SIZEOF_CHAR)
CHECK_TYPE_SIZE("short"         CONDUIT_SIZEOF_SHORT)
CHECK_TYPE_SIZE("int"           CONDUIT_SIZEOF_INT)
CHECK_TYPE_SIZE("long"          CONDUIT_SIZEOF_LONG)

CHECK_TYPE_SIZE("float"         CONDUIT_SIZEOF_FLOAT)
CHECK_TYPE_SIZE("double"        CONDUIT_SIZEOF_DOUBLE)

CHECK_TYPE_SIZE("long long"     CONDUIT_SIZEOF_LONG_LONG)
CHECK_TYPE_SIZE("long float"    CONDUIT_SIZEOF_LONG_FLOAT)
CHECK_TYPE_SIZE("long double"   CONDUIT_SIZEOF_LONG_DOUBLE)

CHECK_TYPE_SIZE("void *"        CONDUIT_SIZEOF_VOID_P)

if(CONDUIT_SIZEOF_LONG_LONG)
    set(CONDUIT_HAS_LONG_LONG 1)
endif()

if(CONDUIT_SIZEOF_LONG_DOUBLE)
    set(CONDUIT_HAS_LONG_DOUBLE 1)
endif()



