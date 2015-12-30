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
# For details, see: http://llnl.github.io/conduit/.
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

from spack import *

import socket
import os
import platform
from os.path import join as pjoin

class UberenvConduit(Package):
    """Spack Based Uberenv Build for Conduit Thirdparty Libs """

    homepage = "http://example.com"

    version('0.1', '8d378ef62dedc2df5db447b029b71200')

    #######################
    # standard spack packages
    #######################
    depends_on("cmake")

    #on osx, build mpich for mpi support
    if "darwin" in platform.system().lower():
        depends_on("mpich")
    
    #######################
    # uberenv custom packages
    #######################

    #######################
    # python
    #######################
    depends_on("python3")
    depends_on("py3-sphinx")
    depends_on("py3-breathe")
    depends_on("py3-numpy")

    depends_on("python")
    depends_on("py-sphinx")
    depends_on("py-breathe")
    depends_on("py-numpy")

    #######################
    # i/o packages
    #######################
    depends_on("szip")
    depends_on("hdf5")
    depends_on("silo")

    def url_for_version(self, version):
        dummy_tar_path =  os.path.abspath(pjoin(os.path.split(__file__)[0]))
        dummy_tar_path = pjoin(dummy_tar_path,"uberenv-conduit.tar.gz")
        url      = "file://" + dummy_tar_path
        return url
        
    def install(self, spec, prefix):
        dest_dir     = env["SPACK_DEBUG_LOG_DIR"]
        c_compiler   = env["SPACK_CC"]
        cpp_compiler = env["SPACK_CXX"]
        f_compiler   = None
        # see if we should enable fortran support
        if "SPACK_FC" in env.keys():
            f_compiler   = env["SPACK_FC"]
        sys_type     = spec.architecture
        if env.has_key("SYS_TYPE"):
            sys_type = env["SYS_TYPE"]
        
        #######################
        # TPL Paths
        #######################
        cmake_exe        = pjoin(spec['cmake'].prefix.bin,"cmake")
        python_exe       = pjoin(spec['python'].prefix.bin,"python")
        sphinx_build_exe = pjoin(spec['python'].prefix.bin,"sphinx-build")
        python3_exe      = pjoin(spec['python3'].prefix.bin,"python3")
        py3_sphinx_build_exe = pjoin(spec['python3'].prefix.bin,"sphinx-build")
        
        #######################
        # Check for MPI
        #######################
        mpicc   = which("mpicc")
        mpicxx  = which("mpicxx")
        mpif90  = which("mpif90")
        mpiexec = which("mpiexec")
        
        print "cmake executable: %s" % cmake_exe
        
        #######################
        # Create host-config
        #######################
        host_cfg_fname = "%s-%s-%s.cmake" % (socket.gethostname(),sys_type,spec.compiler)
        host_cfg_fname = pjoin(dest_dir,host_cfg_fname)

        cfg = open(host_cfg_fname,"w")
        cfg.write("##################################\n")
        cfg.write("# uberenv host-config\n")
        cfg.write("##################################\n")
        cfg.write("# %s-%s\n" % (sys_type,spec.compiler))
        cfg.write("##################################\n\n")
        # show path to cmake for reference
        cfg.write("# cmake from uberenv\n")
        cfg.write("# cmake exectuable path: %s\n\n" % cmake_exe)
        #######################
        # compiler settings
        #######################
        cfg.write("#######\n")
        cfg.write("# using %s compiler spec\n" % spec.compiler)
        cfg.write("#######\n\n")
        cfg.write("# c compiler used by spack\n")
        cfg.write('set(CMAKE_C_COMPILER "%s" CACHE PATH "")\n\n' % c_compiler)
        cfg.write("# cpp compiler used by spack\n")
        cfg.write('set(CMAKE_CXX_COMPILER "%s" CACHE PATH "")\n\n' % cpp_compiler)
        cfg.write("# fortran compiler used by spack\n")
        if not f_compiler is None:
            cfg.write('set(ENABLE_FORTRAN ON CACHE PATH "")\n\n')
            cfg.write('set(CMAKE_Fortran_COMPILER  "%s" CACHE PATH "")\n\n' % f_compiler)
        else:
            cfg.write("# no fortran compiler found\n\n")
            cfg.write('set(ENABLE_FORTRAN OFF CACHE PATH "")\n\n')
        #######################
        #python packages
        #######################
        # python 2
        #######################
        cfg.write("# Enable python module builds\n")
        cfg.write('set(ENABLE_PYTHON ON CACHE PATH "")\n\n')
        cfg.write("# python from uberenv\n")
        cfg.write('set(PYTHON_EXECUTABLE "%s" CACHE PATH "")\n\n' % python_exe)
        cfg.write("# sphinx from uberenv\n")
        cfg.write('set(SPHINX_EXECUTABLE "%s" CACHE PATH "")\n\n' % sphinx_build_exe)
        #######################
        # python 3
        #######################
        cfg.write("# python3 from uberenv\n")
        cfg.write('#set(PYTHON_EXECUTABLE "%s" CACHE PATH "")\n\n' % python3_exe)
        cfg.write("# sphinx from uberenv\n")
        cfg.write('#set(SPHINX_EXECUTABLE "%s" CACHE PATH "")\n\n' % py3_sphinx_build_exe)
        #######################
        # mpi
        #######################
        cfg.write("# MPI Support\n")
        cfg.write('set(ENABLE_MPI ON CACHE PATH "")\n\n')
        if not mpicc is None:
            cfg.write('set(MPI_C_COMPILER  "%s" CACHE PATH "")\n\n' % mpicc.command)
        # we use `mpicc` as `MPI_CXX_COMPILER` b/c we don't want to introduce 
        # linking deps to the MPI C++ libs (we aren't using C++ features of MPI)
        if not mpicxx is None:
            cfg.write('set(MPI_CXX_COMPILER "%s" CACHE PATH "")\n\n' % mpicc.command)
        if not mpif90 is None:
            cfg.write('set(MPI_Fortran_COMPILER "%s" CACHE PATH "")\n\n' % mpif90.command)
        if not mpiexec is None:
            cfg.write('set(MPIEXEC "%s" CACHE PATH "")\n\n' % mpiexec.command)

        #######################
        # i/o packages
        #######################
        cfg.write("# I/O Packages\n\n")
        #######################
        # hdf5
        #######################
        cfg.write("# Enable HDF5 Support in conduit_io\n")
        cfg.write('set(ENABLE_HDF5 ON CACHE PATH "")\n\n')
        cfg.write("# hdf5 from uberenv\n")
        cfg.write('set(HDF5_DIR "%s" CACHE PATH "")\n\n' % spec['hdf5'].prefix)
        #######################
        # silo
        #######################
        cfg.write("# Enable Silo Support in conduit_io\n")
        cfg.write('set(ENABLE_SILO ON CACHE PATH "")\n\n')
        cfg.write("# silo from uberenv\n")
        cfg.write('set(SILO_DIR "%s" CACHE PATH "")\n\n' % spec['silo'].prefix)

        cfg.write("##################################\n")
        cfg.write("# end uberenv host-config\n")
        cfg.write("##################################\n")
        cfg.close()
        
        # place a copy in the spack install dir for the uberenv-conduit package 
        mkdirp(prefix)
        install(host_cfg_fname,prefix)
        print "[result host-config file: %s]" % host_cfg_fname


