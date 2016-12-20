###############################################################################
# Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see: http://software.llnl.gov/conduit/.
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
# CMake Cache Seed file for LLNL BG/Q Systems
#
###############################################################################

set(BUILD_SHARED_LIBS OFF CACHE PATH "")
set(ENABLE_DOCS OFF CACHE PATH "")

set(CMAKE_C_COMPILER "/usr/local/bin/bgxlc_r" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/usr/local/bin/bgxlc++_r" CACHE PATH "")

set(ENABLE_FORTRAN OFF CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/local/bin/bgxlf2003_r" CACHE PATH "")

###############################################################################
# MPI Settings
###############################################################################
set(ENABLE_MPI ON CACHE PATH "")
#
# Explicitly set the mpi include paths and libs for BGQ.
#
# Note: We don't use compiler wrappers on BGQ, b/c CMake will try to link the
# shared variants of the mpi libs instead of the static ones
###############################################################################
set(MPI_INCLUDE_PATHS "/bgsys/drivers/V1R2M4/ppc64/comm/include;/bgsys/drivers/V1R2M4/ppc64/comm/lib/xl;/bgsys/drivers/V1R2M4/ppc64;/bgsys/drivers/V1R2M4/ppc64/comm/sys/include;/bgsys/drivers/V1R2M4/ppc64/spi/include;/bgsys/drivers/V1R2M4/ppc64/spi/include/kernel/cnk" )


set(MPI_LIBS "/bgsys/drivers/V1R2M4/ppc64/comm/lib/libmpich-xl.a;/bgsys/drivers/V1R2M4/ppc64/comm/lib/libopa-xl.a;/bgsys/drivers/V1R2M4/ppc64/comm/lib/libmpl-xl.a;/bgsys/drivers/V1R2M4/ppc64/comm/lib/libpami-gcc.a;/bgsys/drivers/V1R2M4/ppc64/spi/lib/libSPI.a;/bgsys/drivers/V1R2M4/ppc64/spi/lib/libSPI_cnk.a;rt;pthread;stdc++;pthread")


set(MPI_C_INCLUDE_PATH ${MPI_INCLUDE_PATHS} CACHE PATH "")
set(MPI_C_LIBRARIES ${MPI_LIBS} CACHE PATH "")

set(MPI_CXX_INCLUDE_PATH  ${MPI_INCLUDE_PATHS} CACHE PATH "")
set(MPI_CXX_LIBRARIES ${MPI_LIBS} CACHE PATH "")


