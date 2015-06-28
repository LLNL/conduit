###############################################################################
# Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
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
#
# CMake Cache Seed file for naples (Cyrus' laptop)
#

# use clang compilers
set(CMAKE_C_COMPILER "clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "clang++" CACHE PATH "")

set(ENABLE_FORTRAN ON CACHE PATH "")

# Enable mpi for conduit-mpi
set(ENABLE_MPI ON CACHE PATH "")

set(MPIEXEC /Users/harrison37/Work/masonry/build-mb-2.9.1-darwin-10.9-x86_64/thirdparty_shared/visit/mpich/3.0.4/darwin-x86_64/bin/mpiexec CACHE PATH "")

set(MPI_CC_COMPILER /Users/harrison37/Work/masonry/build-mb-2.9.1-darwin-10.9-x86_64/thirdparty_shared/visit/mpich/3.0.4/darwin-x86_64/bin/mpicc CACHE PATH "")

set(MPI_CXX_COMPILER /Users/harrison37/Work/masonry/build-mb-2.9.1-darwin-10.9-x86_64/thirdparty_shared/visit/mpich/3.0.4/darwin-x86_64/bin/mpicc CACHE PATH "")

#######
# uberenv host-config for conduit
#######
# cmake from uberenv
# cmake exectuable path: /Users/harrison37/Work/conduit/uberenv_libs/spack/opt/macosx_10.9_x86_64/gcc@4.2.1/cmake@3.0.2/bin/cmake

# Enable python module builds
set(ENABLE_PYTHON ON CACHE PATH "")

# python from uberenv
set(PYTHON_EXECUTABLE "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/macosx_10.9_x86_64/gcc@4.2.1/python@2.7.8/bin/python" CACHE PATH "")

# sphinx from uberenv
set(SPHINX_EXECUTABLE "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/macosx_10.9_x86_64/gcc@4.2.1/python@2.7.8/bin/sphinx-build" CACHE PATH "")

# I/O Packages

# Enable Silo Support in conduit_io
set(ENABLE_SILO ON CACHE PATH "")

# szip from uberenv
set(SZIP_DIR "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/macosx_10.9_x86_64/gcc@4.2.1/szip@2.1" CACHE PATH "")

# hdf5 from uberenv
set(HDF5_DIR "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/macosx_10.9_x86_64/gcc@4.2.1/hdf5@1.8.7-83ab1427" CACHE PATH "")

# silo from uberenv
set(SILO_DIR "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/macosx_10.9_x86_64/gcc@4.2.1/silo@4.10.1-f5eccf8d" CACHE PATH "")



