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
# For details, see: http://cyrush.github.io/conduit.
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
from os.path import join as pjoin

class UberenvConduit(Package):
    """Spack Based Uberenv Build for Conduit Thirdparty Libs """

    homepage = "http://example.com"

    version('0.1', '8d378ef62dedc2df5db447b029b71200')
    
    # all of these packages are custom
    depends_on("python")
    depends_on("py-sphinx")
    depends_on("py-breathe")
    depends_on("py-numpy")
    depends_on("cmake")
    # i/o packages
    depends_on("szip")
    depends_on("hdf5")
    depends_on("silo")

    def url_for_version(self, version):
        print __file__
        dummy_tar_path =  os.path.abspath(pjoin(os.path.split(__file__)[0]))
        dummy_tar_path = pjoin(dummy_tar_path,"uberenv-conduit.tar.gz")
        url      = "file://" + dummy_tar_path
        return url
        
    def install(self, spec, prefix):
        dest_dir     = env["SPACK_DEBUG_LOG_DIR"]
        c_compiler   = env["SPACK_CC"]
        cpp_compiler = env["SPACK_CXX"]
        sys_type     = spec.architecture
        if env.has_key("SYS_TYPE"):
            sys_type = env["SYS_TYPE"]
        cmake_exe        = pjoin(spec['cmake'].prefix.bin,"cmake")
        python_exe       = pjoin(spec['python'].prefix.bin,"python")
        sphinx_build_exe = pjoin(spec['python'].prefix.bin,"sphinx-build")
        # TODO: better name (use sys-type and compiler name ?)
        print "cmake executable: %s" % cmake_exe
        cfg = open(pjoin(dest_dir,"%s.cmake" % socket.gethostname()),"w")
        cfg.write("##################################\n")
        cfg.write("# uberenv host-config for conduit\n")
        cfg.write("##################################\n")
        cfg.write("# %s-%s\n" % (sys_type,spec.compiler))
        cfg.write("##################################\n\n")
        # show path to cmake for reference
        cfg.write("# cmake from uberenv\n")
        cfg.write("# cmake exectuable path: %s\n\n" % cmake_exe)
        # compiler settings
        cfg.write("#######\n")
        cfg.write("# using %s compiler spec\n" % spec.compiler)
        cfg.write("#######\n\n")
        cfg.write("# c compiler used by spack\n")
        cfg.write('set(CMAKE_C_COMPILER "%s" CACHE PATH "")\n\n' % c_compiler)
        cfg.write("# cpp compiler used by spack\n")
        cfg.write('set(CMAKE_CXX_COMPILER "%s" CACHE PATH "")\n\n' % cpp_compiler)
        #python packages
        cfg.write("# Enable python module builds\n")
        cfg.write('set(ENABLE_PYTHON ON CACHE PATH "")\n\n')
        cfg.write("# python from uberenv\n")
        cfg.write('set(PYTHON_EXECUTABLE "%s" CACHE PATH "")\n\n' % python_exe)
        cfg.write("# sphinx from uberenv\n")
        cfg.write('set(SPHINX_EXECUTABLE "%s" CACHE PATH "")\n\n' % sphinx_build_exe)
        # i/o packages
        cfg.write("# I/O Packages\n\n")
        cfg.write("# Enable Silo Support in conduit_io\n")
        cfg.write('set(ENABLE_SILO ON CACHE PATH "")\n\n')
        cfg.write("# szip from uberenv\n")
        cfg.write('set(SZIP_DIR "%s" CACHE PATH "")\n\n' % spec['szip'].prefix)
        cfg.write("# hdf5 from uberenv\n")
        cfg.write('set(HDF5_DIR "%s" CACHE PATH "")\n\n' % spec['hdf5'].prefix)
        cfg.write("# silo from uberenv\n")
        cfg.write('set(SILO_DIR "%s" CACHE PATH "")\n\n' % spec['silo'].prefix)
        cfg.close()
        