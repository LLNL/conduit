#!/bin/bash
rm -rf build-debug
mkdir build-debug
cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug ../src 
cd ../
# also create an xcode build for debugging on osx
if [ "$TERM_PROGRAM" = "Apple_Terminal" ]; then
    rm -rf build-debug-xcode
    mkdir build-debug-xcode
    cd build-debug-xcode
    cmake -G Xcode -DCMAKE_BUILD_TYPE=Debug ../src 
    cd ../
fi