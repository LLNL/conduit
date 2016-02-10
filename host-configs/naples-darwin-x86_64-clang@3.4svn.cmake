##################################
# uberenv host-config
##################################
# darwin-x86_64-clang@3.4svn
##################################

# cmake from uberenv
# cmake exectuable path: /Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/cmake-3.4.0-hnxlu7d6xlhneergdveml5jtybsg6zh7/bin/cmake

#######
# using clang@3.4svn compiler spec
#######

# c compiler used by spack
set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE PATH "")

# cpp compiler used by spack
set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE PATH "")

# fortran compiler used by spack
set(ENABLE_FORTRAN ON CACHE PATH "")

set(CMAKE_Fortran_COMPILER  "/sw/bin/gfortran" CACHE PATH "")

# Enable python module builds
set(ENABLE_PYTHON ON CACHE PATH "")

# python from uberenv
set(PYTHON_EXECUTABLE "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/python-2.7.8-dgjtqtwr657ptot6eqp3yfwp3cqns7yx/bin/python" CACHE PATH "")

# sphinx from uberenv
set(SPHINX_EXECUTABLE "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/python-2.7.8-dgjtqtwr657ptot6eqp3yfwp3cqns7yx/bin/sphinx-build" CACHE PATH "")

# python3 from uberenv
#set(PYTHON_EXECUTABLE "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/python3-3.4.3-urgetnenpwtvifkdeftodvlpp7jmd3xa/bin/python3" CACHE PATH "")

# sphinx from uberenv
#set(SPHINX_EXECUTABLE "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/python3-3.4.3-urgetnenpwtvifkdeftodvlpp7jmd3xa/bin/sphinx-build" CACHE PATH "")

# MPI Support
set(ENABLE_MPI ON CACHE PATH "")

set(MPI_C_COMPILER  "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/mpich-3.1.4-ft7znm6qg5zxfjite6keb67z36igeaoq/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/mpich-3.1.4-ft7znm6qg5zxfjite6keb67z36igeaoq/bin/mpicc" CACHE PATH "")

set(MPI_Fortran_COMPILER "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/mpich-3.1.4-ft7znm6qg5zxfjite6keb67z36igeaoq/bin/mpif90" CACHE PATH "")

set(MPIEXEC "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/mpich-3.1.4-ft7znm6qg5zxfjite6keb67z36igeaoq/bin/mpiexec" CACHE PATH "")

# I/O Packages

# hdf5 from uberenv
set(HDF5_DIR "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/hdf5-1.8.16-remxmdztkcscdmxicoxfwzpq2jj6zrw3" CACHE PATH "")

# silo from uberenv
set(SILO_DIR "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/silo-4.10.1-esuslfjr676bqf4ptgtsxoqqzvvvxpwd" CACHE PATH "")

##################################
# end uberenv host-config
##################################
