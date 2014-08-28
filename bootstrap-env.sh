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


#
# file: bootstrap-env.sh
#

#
# Takes you from zero to a Python + CMake env for conduit on OSX and linux
#

export PY_VERSION="2.7.6"

function info
{
    echo "$@"
}

function warn
{
    info "WARNING: $@"
}

function error
{
    info "ERROR: $@"
    if [ x"${BASH_SOURCE[0]}" == x"$0" ] ; then
        exit 1
    else
        kill -SIGINT $$
    fi
}

function check
{
    if [[ $1 != 0 ]] ; then
        error " !Last step failed!"
    fi
}

function download
{
    if [[ -e $2 ]] ; then
        info "[Found: $2 -- Skipping download]"
        return 0
    fi
    info "[NOT Found: $2  -- Downloading from $1]"
    WGET_TEST=$(which wget)
    if [[ $WGET_TEST == "" ]] ; then
        curl -ksfLO $1/$2
    else
        wget --no-check-certificate $1/$2
    fi
}

function set_install_path
{
    export START_DIR=`pwd`
    export BUILD_DIR=$1/_build
    export LOGS_DIR=$BUILD_DIR/logs
    export PY_ROOT=$1/python
    export PY_PREFIX=$PY_ROOT/$PY_VERSION
    export PY_EXE=$PY_PREFIX/bin/python
    export PIP_EXE=$PY_PREFIX/bin/pip
    export CMAKE_PREFIX=$1/cmake
}

function check_python_install
{
    if [[ -e $1/python/$PY_VERSION/bin/python ]] ; then
        return 0
    else
        return 1
    fi
}

function check_osx
{
    $PY_EXE <<END
import sys
import platform
if sys.platform.count("darwin") > 0:
    sys.exit(0)
else:
    sys.exit(-1)

END
    return $?
}

function check_osx_10_8
{
    $PY_EXE <<END
import sys
import platform
osx_ml = False
if sys.platform.count("darwin") > 0:
    if platform.mac_ver()[0].count("10.8") > 0:
        osx_ml = True

if osx_ml:
    sys.exit(0)
else:
    sys.exit(-1)

END
    return $?
}

function check_cmake_install
{
    if [[ -e $1/cmake/bin/cmake ]] ; then
        return 0
    else
        return 1
    fi
}

function build_cmake
{
    info "================================="
    info "Setting up CMake 2.8.12.2"
    info "================================="
    info "[Target Prefix: $CMAKE_PREFIX]"
    cd $BUILD_DIR
    download http://www.cmake.org/files/v2.8/ cmake-2.8.12.2.tar.gz
    rm -rf cmake-2.8.12
    info "[Inflating: cmake-2.8.12.2.tar.gz]"
    tar -xzf cmake-2.8.12.2.tar.gz
    cd cmake-2.8.12.2
    info "[Configuring CMake]"
    ./configure --prefix=$CMAKE_PREFIX &> ../logs/cmake_configure.txt
    check $?
    info "[Building CMake]"
    make -j 4 &> ../logs/cmake_build.txt
    check $?
    info "[Installing CMake]"
    make install &> ../logs/cmake_install.txt
    check $?
    cd $START_DIR
}



function bootstrap_python
{
    mkdir $PY_ROOT
    mkdir $PY_PREFIX

    info "================================="
    info "Bootstraping Python $PY_VERSION"
    info "================================="
    info "[Target Prefix: $PY_PREFIX]"
    cd $BUILD_DIR
    download http://www.python.org/ftp/python/$PY_VERSION Python-$PY_VERSION.tgz
    rm -rf Python-$PY_VERSION
    info "[Inflating: Python-$PY_VERSION.tgz]"
    tar -xzf Python-$PY_VERSION.tgz
    cd Python-$PY_VERSION
    info "[Configuring Python]"
    mkdir -p ${PY_PREFIX}/lib/ z
    ./configure --enable-shared --prefix=$PY_PREFIX LDFLAGS='-Wl,-rpath,${PY_PREFIX}/lib/ -pthread'  &> ../logs/python_configure.txt

    check $?
    info "[Building Python]"
    make -j 4 &> ../logs/python_build.txt
    check $?
    info "[Installing Python]"
    make install &> ../logs/python_install.txt
    check $?

    cd $START_DIR
}

function bootstrap_modules
{
    # bootstrap pip
    info "================================="
    info "Bootstraping base modules"
    info "================================="

    cd $BUILD_DIR
    download https://pypi.python.org/packages/source/s/setuptools/ setuptools-1.3.tar.gz
    rm -rf setuptools-1.3
    info "[Inflating: setuptools-1.3.tar.gz]"
    tar -xzf setuptools-1.3.tar.gz
    cd setuptools-1.3
    info "[Building setuptools]"
    $PY_EXE setup.py build &> ../logs/setuptools_build.txt
    check $?
    info "[Installing setuptools]"
    $PY_EXE setup.py install &> ../logs/setuptools_install.txt
    check $?


    cd $BUILD_DIR
    download http://pypi.python.org/packages/source/p/pip pip-1.2.1.tar.gz
    rm -rf pip-1.2.1
    info "[Inflating: pip-1.2.1.tar.gz]"
    tar -xzf pip-1.2.1.tar.gz
    cd pip-1.2.1
    info "[Building pip]"
    $PY_EXE setup.py build &> ../logs/pip_build.txt
    check $?
    info "[Installing pip]"
    $PY_EXE setup.py install &> ../logs/pip_install.txt
    check $?

    cd $START_DIR
}


function build_python_modules
{
    # only install readline on osx
    if check_osx; then
        $PIP_EXE install readline
    fi;
    # numpy and cython
    $PIP_EXE install numpy
}


function main
{
    DEST=`pwd`/libs
    mkdir -p $DEST
    set_install_path $DEST
    
    mkdir $BUILD_DIR
    mkdir $LOGS_DIR
    
    check_cmake_install $DEST
    if [[ $? == 0 ]] ; then
        info "[Found: CMake @ $CMAKE_PREFIX -- Skipping build]"
    else
        build_cmake
    fi
    
    # Only add to PATH if `which cmake` isn't our CMake
    CMAKE_CURRENT=`which cmake`
    if [[ "$CMAKE_CURRENT" != "$CMAKE_PREFIX/bin/cmake" ]] ; then
        export PATH=$CMAKE_PREFIX/bin:$PATH
    fi
    
    info "[Active CMake:" `which cmake` "]"
    # we no longer rely on numpy for bitwidth types
    return
    
    check_python_install $DEST
    if [[ $? == 0 ]] ; then
        info "[Found: Python $PY_VERSION @ $DEST/$PY_VERSION/bin/python -- Skipping build]"
    else
        bootstrap_python
        bootstrap_modules
    fi
    # Only add to PATH if `which python` isn't our Python
    PY_CURRENT=`which python`
    if [[ "$PY_CURRENT" != "$PY_PREFIX/bin/python" ]] ; then
        export PATH=$PY_PREFIX/bin:$PATH
    fi

    build_python_modules

    info "[Active CMake:" `which cmake` "]"
    info "[Active Python:" `which python` "]"

}

main
