#!/bin/bash
###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.


#
# file: bootstrap-env.sh
#

#
# Takes you from zero to an env  with TPLS needed to develop conduit on OSX
# and linux.
#

export ALL_ARGS="$@"

function info
{
    echo "$@"
}

function uberenv
{
    python scripts/uberenv/uberenv.py "$ALL_ARGS"
}

function main
{
    uberenv

    BOOSTRAP_CWD=`pwd`
    SPACK_CMAKE_PREFIX=`ls -d $BOOSTRAP_CWD/uberenv_libs/spack/opt/spack/*/*/cmake*`
    SPACK_CMAKE=`ls $SPACK_CMAKE_PREFIX/bin/cmake`

    # Only add to PATH if `which cmake` isn't our CMake
    CMAKE_CURRENT=`which cmake`
    if [[ "$CMAKE_CURRENT" != "$SPACK_CMAKE" ]] ; then
        export PATH=$SPACK_CMAKE_PREFIX/bin:$PATH
    fi
    
    info "[Active CMake:" `which cmake` "]"
}

main
