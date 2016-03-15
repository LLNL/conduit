##################################
# uberenv host-config
##################################
# darwin-x86_64-clang@3.4svn
##################################

# cmake from uberenv
# cmake exectuable path: /Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/cmake-3.3.1-4u3hrpeigz5i5u5tzbqwlcfyc2vskauh/bin/cmake

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
set(PYTHON_EXECUTABLE "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/python-2.7.11-jswmndoc7h6ruppd6atok7hvx4dn7h6j/bin/python" CACHE PATH "")

# sphinx from uberenv
set(SPHINX_EXECUTABLE "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/python-2.7.11-jswmndoc7h6ruppd6atok7hvx4dn7h6j/bin/sphinx-build" CACHE PATH "")

# python3 from uberenv
#set(PYTHON_EXECUTABLE "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/python3-3.5.1-d4u6mvmabzpv2zw6icqp55awuuxiib6b/bin/python3" CACHE PATH "")

# sphinx from uberenv
#set(SPHINX_EXECUTABLE "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/python3-3.5.1-d4u6mvmabzpv2zw6icqp55awuuxiib6b/bin/sphinx-build" CACHE PATH "")

# MPI Support
set(ENABLE_MPI ON CACHE PATH "")

set(MPI_C_COMPILER  "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/mpich-3.2-5m2s3c3ugo6jqt4bz7nci3nfh4aamzfu/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/mpich-3.2-5m2s3c3ugo6jqt4bz7nci3nfh4aamzfu/bin/mpicc" CACHE PATH "")

set(MPI_Fortran_COMPILER "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/mpich-3.2-5m2s3c3ugo6jqt4bz7nci3nfh4aamzfu/bin/mpif90" CACHE PATH "")

set(MPIEXEC "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/mpich-3.2-5m2s3c3ugo6jqt4bz7nci3nfh4aamzfu/bin/mpiexec" CACHE PATH "")

# I/O Packages

# hdf5 from uberenv
set(HDF5_DIR "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/hdf5-1.8.16-cnnhzbzr3ssaplydmehw7abbrpmpvb6q" CACHE PATH "")

# silo from uberenv
set(SILO_DIR "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/silo-4.10.1-w6uzqebhfd4oy36bjjucd7vd3fhlu5i3" CACHE PATH "")

##################################
# end uberenv host-config
##################################
