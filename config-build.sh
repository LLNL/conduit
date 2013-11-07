#!/bin/bash
rm -rf build-debug
mkdir build-debug
cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug ../src 
cd ../