# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
###############################################################################
#
# CMake Cache Seed file for LLNL BG/Q Systems
#
###############################################################################

set(BUILD_SHARED_LIBS OFF CACHE PATH "")
set(ENABLE_DOCS OFF CACHE PATH "")

set(CMAKE_C_COMPILER "/usr/local/tools/toolchain-4.8.4/gnu-linux-4.8.4/bin/powerpc64-bgq-linux-gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/usr/local/tools/toolchain-4.8.4/gnu-linux-4.8.4/bin/powerpc64-bgq-linux-g++" CACHE PATH "")

set(ENABLE_FORTRAN ON CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/local/tools/toolchain-4.8.4/gnu-linux-4.8.4/bin/powerpc64-bgq-linux-gfortran" CACHE PATH "")

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

set(MPI_LIBS "/bgsys/drivers/V1R2M4/ppc64/comm/lib/libmpich-gcc.a;/bgsys/drivers/V1R2M4/ppc64/comm/lib/libopa-gcc.a;/bgsys/drivers/V1R2M4/ppc64/comm/lib/libmpl-gcc.a;/bgsys/drivers/V1R2M4/ppc64/comm/lib/libpami-gcc.a;/bgsys/drivers/V1R2M4/ppc64/spi/lib/libSPI.a;/bgsys/drivers/V1R2M4/ppc64/spi/lib/libSPI_cnk.a;rt;pthread;stdc++;pthread")

set(MPI_INCLUDE_PATHS "/bgsys/drivers/V1R2M4/ppc64/comm/include;/bgsys/drivers/V1R2M4/ppc64/comm/lib/gnu;/bgsys/drivers/V1R2M4/ppc64;/bgsys/drivers/V1R2M4/ppc64/comm/sys/include;/bgsys/drivers/V1R2M4/ppc64/spi/include;/bgsys/drivers/V1R2M4/ppc64/spi/include/kernel/cnk" )

set(MPI_C_INCLUDE_PATH ${MPI_INCLUDE_PATHS} CACHE PATH "")
set(MPI_C_LIBRARIES ${MPI_LIBS} CACHE PATH "")

set(MPI_CXX_INCLUDE_PATH  ${MPI_INCLUDE_PATHS} CACHE PATH "")
set(MPI_CXX_LIBRARIES ${MPI_LIBS} CACHE PATH "")



