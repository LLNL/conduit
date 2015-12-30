##################################
# uberenv host-config
##################################
# chaos_5_x86_64_ib-gcc@4.9.3
##################################

# cmake from uberenv
# cmake exectuable path: /usr/gapps/conduit/thirdparty_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/cmake-3.4.0-bupmqgyltb5wenlmm57ze3h43uqg5lx5/bin/cmake

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
set(PYTHON_EXECUTABLE "/usr/gapps/conduit/thirdparty_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/python-2.7.8-wxs6x4molggy6niut2czpjfhvhmrscj5/bin/python" CACHE PATH "")

# sphinx from uberenv
set(SPHINX_EXECUTABLE "/usr/gapps/conduit/thirdparty_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/python-2.7.8-wxs6x4molggy6niut2czpjfhvhmrscj5/bin/sphinx-build" CACHE PATH "")

# python3 from uberenv
#set(PYTHON_EXECUTABLE "/usr/gapps/conduit/thirdparty_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/python3-3.4.3-cy7dff7xqjqon57u56h5atc6kkfxcilc/bin/python3" CACHE PATH "")

# sphinx from uberenv
#set(SPHINX_EXECUTABLE "/usr/gapps/conduit/thirdparty_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/python3-3.4.3-cy7dff7xqjqon57u56h5atc6kkfxcilc/bin/sphinx-build" CACHE PATH "")

# MPI Support
set(ENABLE_MPI ON CACHE PATH "")

set(MPI_C_COMPILER  "/usr/local/bin/mpicc" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/local/bin/mpif90" CACHE PATH "")

# I/O Packages

# Enable HDF5 Support in conduit_io
set(ENABLE_HDF5 ON CACHE PATH "")

# hdf5 from uberenv
set(HDF5_DIR "/usr/gapps/conduit/thirdparty_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/hdf5-1.8.16-rmyziv2il5qhgnpntj2dsr7qusrn7ual" CACHE PATH "")

# Enable Silo Support in conduit_io
set(ENABLE_SILO ON CACHE PATH "")

# silo from uberenv
set(SILO_DIR "/usr/gapps/conduit/thirdparty_libs/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/silo-4.10.1-seumccglv6ja5dv2myyxatsjhterselh" CACHE PATH "")

##################################
# end uberenv host-config
##################################


# Enable mpi for conduit-mpi
set(ENABLE_MPI ON CACHE PATH "")
set(MPIEXEC /usr/bin/srun CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG -p pdebug -n CACHE PATH "")


