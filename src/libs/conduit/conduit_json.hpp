#ifndef CONDUIT_JSON_HPP
#define CONDUIT_JSON_HPP

#ifdef USE_YYJSON
  #include "conduit_yyjson.h"
  #define conduit_json conduit_yyjson
#else
  #include "rapidjson/document.h"
  #include "rapidjson/error/en.h"
  #define conduit_json conduit_rapidjson
#endif

#endif
