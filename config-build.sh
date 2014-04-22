#!/bin/bash
rm -rf build-debug install-debug
mkdir build-debug
mkdir install-debug
cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../install-debug ../src 
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