#!/bin/bash
cd build-debug
make -j 4 && make test
cd ..