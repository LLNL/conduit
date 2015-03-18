#!/bin/bash
###############################################################################
# Copyright (c) 2014, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see https://lc.llnl.gov/conduit/.
# 
# Please also read conduit/LICENSE
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.
# 
###############################################################################


#
# file: bootstrap-env.sh
#

#
# Takes you from zero to a Python + CMake env for conduit on OSX and linux
#

export PY_VERSION="2.7.9"

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
    info "Setting up CMake 3.1.3"
    info "================================="
    info "[Target Prefix: $CMAKE_PREFIX]"
    cd $BUILD_DIR
    download http://www.cmake.org/files/v3.1/ cmake-3.1.3.tar.gz
    rm -rf cmake-3.0.2
    info "[Inflating: cmake-3.1.3.tar.gz]"
    tar -xzf cmake-3.1.3.tar.gz
    cd cmake-3.1.3
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
    mkdir -p ${PY_PREFIX}/lib/
    ./configure --enable-shared --prefix=$PY_PREFIX &> ../logs/python_configure.txt
    # LDFLAGS='-Wl,-rpath,${PY_PREFIX}/lib/ -pthread'  

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
    download https://pypi.python.org/packages/source/s/setuptools/ setuptools-14.0.tar.gz
    rm -rf setuptools-14
    info "[Inflating: setuptools-14.0.tar.gz]"
    tar -xzf setuptools-14.0.tar.gz
    cd setuptools-14.0
    info "[Building setuptools]"
    $PY_EXE setup.py build &> ../logs/setuptools_build.txt
    check $?
    info "[Installing setuptools]"
    $PY_EXE setup.py install &> ../logs/setuptools_install.txt
    check $?

    cd $BUILD_DIR
    
    download http://pypi.python.org/packages/source/p/pip pip-6.0.8.tar
    rm -rf pip-6.0.8
    info "[Inflating: pip-6.0.8.tar.gz]"
    tar -xzf pip-6.0.8.tar.gz
    cd pip-6.0.8
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
    # numpy
    $PIP_EXE install numpy
    # sphinx
    $PIP_EXE install sphinx
    # breathe
    $PIP_EXE install pip breathe 
    # rtd theme
    $PIP_EXE install pip install https://github.com/snide/sphinx_rtd_theme/archive/master.zip


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
