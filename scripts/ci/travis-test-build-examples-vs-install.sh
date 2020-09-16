#!/bin/bash
###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

set -ev

if [ "${ENABLE_COVERAGE}" == "ON" ]; then
    echo "skipping examples vs install tests (ENABLE_COVERAGE=ON)"
    exit 0
fi

##########################################################
# test our examples that demo using an installed conduit
##########################################################

##########################################
# using with cmake example
##########################################
cd ${TRAVIS_BUILD_DIR}/travis-debug-install/examples/conduit/using-with-cmake
mkdir build
cd build
cmake -DCONDUIT_DIR=${TRAVIS_BUILD_DIR}/travis-debug-install ../
make
./conduit_example
##########################################
# using with make example
##########################################
export RUN_LIB_PATH="${TRAVIS_BUILD_DIR}/travis-debug-install/lib/"


cd ${TRAVIS_BUILD_DIR}/travis-debug-install/examples/conduit/using-with-make

if [ "${ENABLE_HDF5}" == "ON" ]; then
    # find spack installed HDF5
    export HDF5_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/hdf5*`
    export RUN_LIB_PATH="${RUN_LIB_PATH}:${HDF5_DIR}/lib"
fi

if [ "${ENABLE_SILO}" == "ON" ]; then
    # find spack installed Silo
    export SILO_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/silo*`
    export RUN_LIB_PATH="${RUN_LIB_PATH}:${SILO_DIR}/lib"
fi

if [ "${ENABLE_ADIOS}" == "ON" ]; then
    # find spack installed ADIOS
    export ADIOS_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/adios*`
    export ZFP_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/zfp*`
    export LZ4_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/lz4*`
    export BLOSC_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/c-blosc*`
    export SZ_DIR=`ls -d ${TRAVIS_BUILD_DIR}/uberenv_libs/spack/opt/spack/*/*/sz*`
    export RUN_LIB_PATH="${RUN_LIB_PATH}:${ADIOS_DIR}/lib:${ZFP_DIR}/lib:${LZ4_DIR}/lib:${BLOSC_DIR}/lib:${SZ_DIR}/lib"
fi

env CXX=${COMPILER_CXX} CONDUIT_DIR=${TRAVIS_BUILD_DIR}/travis-debug-install make
env LD_LIBRARY_PATH=${RUN_LIB_PATH} ./conduit_example

