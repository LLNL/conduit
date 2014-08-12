################################
# Checks for type sizes
################################

INCLUDE (CheckTypeSize)

CHECK_TYPE_SIZE("boolean"       SIZEOF_BOOLEAN)
CHECK_TYPE_SIZE("char"          SIZEOF_CHAR)
CHECK_TYPE_SIZE("short"         SIZEOF_SHORT)
CHECK_TYPE_SIZE("int"           SIZEOF_INT)
CHECK_TYPE_SIZE("long"          SIZEOF_LONG)

CHECK_TYPE_SIZE("float"         SIZEOF_FLOAT)
CHECK_TYPE_SIZE("double"        SIZEOF_DOUBLE)

CHECK_TYPE_SIZE("long long"     SIZEOF_LONG_LONG)
CHECK_TYPE_SIZE("long float"    SIZEOF_LONG_FLOAT)
CHECK_TYPE_SIZE("long double"   SIZEOF_LONG_DOUBLE)

CHECK_TYPE_SIZE("void *"        SIZEOF_VOID_P)
