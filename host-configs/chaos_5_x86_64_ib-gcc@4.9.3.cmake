##################################
# uberenv host-config
##################################
# chaos_5_x86_64_ib-gcc@4.9.3
##################################

# cmake from uberenv
# cmake exectuable path: /usr/gapps/conduit/thirdparty_libs/stable/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/cmake-3.3.1-3gc4unffj5rqcq35gecg4wv3roecpldt/bin/cmake

#######
# using gcc@4.9.3 compiler spec
#######

# c compiler used by spack
set(CMAKE_C_COMPILER "/usr/apps/gnu/4.9.3/bin/gcc" CACHE PATH "")

# cpp compiler used by spack
set(CMAKE_CXX_COMPILER "/usr/apps/gnu/4.9.3/bin/g++" CACHE PATH "")

# fortran compiler used by spack
set(ENABLE_FORTRAN ON CACHE PATH "")

set(CMAKE_Fortran_COMPILER  "/usr/apps/gnu/4.9.3/bin/gfortran" CACHE PATH "")

# Enable python module builds
set(ENABLE_PYTHON ON CACHE PATH "")

# python from uberenv
set(PYTHON_EXECUTABLE "/usr/gapps/conduit/thirdparty_libs/stable/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/python-2.7.11-eujx7frnxd5vpwolmye2fzq4tcylnbnv/bin/python" CACHE PATH "")

# sphinx from uberenv
set(SPHINX_EXECUTABLE "/usr/gapps/conduit/thirdparty_libs/stable/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/python-2.7.11-eujx7frnxd5vpwolmye2fzq4tcylnbnv/bin/sphinx-build" CACHE PATH "")

# python3 from uberenv
#set(PYTHON_EXECUTABLE "/usr/gapps/conduit/thirdparty_libs/stable/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/python3-3.5.1-bbbluvapi4elzb76vqqriuwq3y7ldqkk/bin/python3" CACHE PATH "")

# sphinx from uberenv
#set(SPHINX_EXECUTABLE "/usr/gapps/conduit/thirdparty_libs/stable/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/python3-3.5.1-bbbluvapi4elzb76vqqriuwq3y7ldqkk/bin/sphinx-build" CACHE PATH "")

# MPI Support
set(ENABLE_MPI ON CACHE PATH "")

set(MPI_C_COMPILER  "/usr/local/bin/mpicc" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/local/bin/mpif90" CACHE PATH "")

# I/O Packages

# hdf5 from uberenv
set(HDF5_DIR "/usr/gapps/conduit/thirdparty_libs/stable/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/hdf5-1.8.16-msbowehgkgvhlnl62fy6tb7bvefbr7h4" CACHE PATH "")

# silo from uberenv
set(SILO_DIR "/usr/gapps/conduit/thirdparty_libs/stable/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/silo-4.10.1-jnuhe4xm3vtwq4mevsobhahlriuqafrg" CACHE PATH "")

##################################
# end uberenv host-config
##################################

# Extra MPI Settings for Chaos
set(MPIEXEC /usr/bin/srun CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG -n CACHE PATH "")


