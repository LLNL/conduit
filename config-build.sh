#!/bin/bash
#############################################################################
# Copyright (c) 2014, Lawrence Livermore National Security, LLC
# Produced at the Lawrence Livermore National Laboratory. 
# 
# All rights reserved.
# 
# This source code cannot be distributed without further review from 
# Lawrence Livermore National Laboratory.
#############################################################################

rm -rf build-debug install-debug
mkdir build-debug
mkdir install-debug

cd build-debug
export CMAKE_OPTS="$CMAKE_OPTS -DCMAKE_BUILD_TYPE=Debug"
export CMAKE_OPTS="$CMAKE_OPTS -DCMAKE_INSTALL_PREFIX=../install-debug"
if [ "$TERM_PROGRAM" = "Apple_Terminal" ]; then
    export CMAKE_OPTS="$CMAKE_OPTS -DCMAKE_C_COMPILER=clang"
    export CMAKE_OPTS="$CMAKE_OPTS -DCMAKE_CXX_COMPILER=clang++"
fi

export HOST_CONFIG=../host-configs/`hostname`.cmake
echo "Looking for host-config file: $HOST_CONFIG"
if [[ -e  "$HOST_CONFIG" ]]; then
    echo "FOUND: $HOST_CONFIG"
    export CMAKE_OPTS="$CMAKE_OPTS -C $HOST_CONFIG"
else
    echo "MISSING: $HOST_CONFIG"
fi
    

echo "cmake $CMAKE_OPTS ../src"
cmake  $CMAKE_OPTS \
       ../src 
cd ../
# add extended builds when an argument is passed
if [ $# -ge 1 ]; then
    # also create an xcode build for debugging on osx
    if [ "$TERM_PROGRAM" = "Apple_Terminal" ]; then
        rm -rf build-debug-xcode
        mkdir build-debug-xcode
        cd build-debug-xcode
        cmake -G Xcode -DCMAKE_BUILD_TYPE=Debug ../src 
        cd ../
    fi
fi