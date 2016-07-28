##################################
# uberenv host-config
##################################
# darwin-x86_64-clang@3.4svn
##################################

# cmake from uberenv
# cmake exectuable path: /Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/cmake-3.3.1-derruj5rkiqrvcxmzh7bwiycyi2kidte/bin/cmake

#######
# using clang@3.4svn compiler spec
#######

# c compiler used by spack
set("CMAKE_C_COMPILER" "/usr/bin/clang" CACHE PATH "")

# cpp compiler used by spack
set("CMAKE_CXX_COMPILER" "/usr/bin/clang++" CACHE PATH "")

# fortran compiler used by spack
# no fortran compiler found

set("ENABLE_FORTRAN" "OFF" CACHE PATH "")

# Enable python module builds
set("ENABLE_PYTHON" "ON" CACHE PATH "")

# python from uberenv
set("PYTHON_EXECUTABLE" "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/python-2.7.11-fmwboy225dowlzykfvloodfx2nuuskd5/bin/python" CACHE PATH "")

# sphinx from uberenv
set("SPHINX_EXECUTABLE" "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/python-2.7.11-fmwboy225dowlzykfvloodfx2nuuskd5/bin/sphinx-build" CACHE PATH "")

# python3 from uberenv
#set("PYTHON_EXECUTABLE" "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/python3-3.5.1-wgiw7dyijzienqtorbsmfyz7xgtvrsnz/bin/python3" CACHE PATH "")

# sphinx from uberenv
#set("SPHINX_EXECUTABLE" "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/python3-3.5.1-wgiw7dyijzienqtorbsmfyz7xgtvrsnz/bin/sphinx-build" CACHE PATH "")

# MPI Support
set("ENABLE_MPI" "ON" CACHE PATH "")

set("MPI_C_COMPILER" "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/mpich-3.2-5m2s3c3ugo6jqt4bz7nci3nfh4aamzfu/bin/mpicc" CACHE PATH "")

set("MPI_CXX_COMPILER" "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/mpich-3.2-5m2s3c3ugo6jqt4bz7nci3nfh4aamzfu/bin/mpicc" CACHE PATH "")

set("MPIEXEC" "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/mpich-3.2-5m2s3c3ugo6jqt4bz7nci3nfh4aamzfu/bin/mpiexec" CACHE PATH "")

# I/O Packages

# hdf5 from uberenv
set("HDF5_DIR" "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/hdf5-1.8.17-pzxksxsggtm6ab72nbymwokfybp66boh" CACHE PATH "")

# silo from uberenv
set("SILO_DIR" "/Users/harrison37/Work/conduit/uberenv_libs/spack/opt/spack/darwin-x86_64/clang-3.4svn/silo-4.10.1-5rka7esxmogsporir2fkg5dkrkgnywtj" CACHE PATH "")

##################################
# end uberenv host-config
##################################
