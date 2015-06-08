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

from spack import *

import socket
from os.path import join as pjoin

class UberenvConduit(Package):
    """Spack Based Uberenv Build for Conduit Thirdparty Libs """

    # TODO: what do we need for DIY mode?
    
    homepage = "http://sphinx-doc.org/"
    url      = "https://pypi.python.org/packages/source/S/Sphinx/Sphinx-1.3.1.tar.gz#md5=8786a194acf9673464c5455b11fd4332"

    version('1.3.1', '8786a194acf9673464c5455b11fd4332')
    
    # all of these packages are custom
    depends_on("python")
    depends_on("py-sphinx")
    depends_on("py-breathe")
    depends_on("py-numpy")
    depends_on("cmake")

    def install(self, spec, prefix):
        dest_dir = env["SPACK_DEBUG_LOG_DIR"]
        cmake_exe        = pjoin(spec['cmake'].prefix.bin,"cmake")
        python_exe       = pjoin(spec['python'].prefix.bin,"python")
        sphinx_build_exe = pjoin(spec['python'].prefix.bin,"sphinx-build")
        # TODO: better name (use sys-type and compiler name ?)
        print "cmake executable: %s" % cmake_exe
        cfg = open(pjoin(dest_dir,"%s.cmake" % socket.gethostname()),"w")
        cfg.write("#######\n")
        cfg.write("# uberenv host-config for conduit\n")
        cfg.write("#######\n")
        cfg.write("# cmake from uberenv\n")
        cfg.write("# cmake exectuable path: %s\n\n" % cmake_exe)
        cfg.write("# python from uberenv\n")
        cfg.write('set(PYTHON_EXECUTABLE "%s" CACHE PATH "")\n\n' % python_exe)
        cfg.write("# sphinx from uberenv\n")
        cfg.write('set(SPHINX_EXECUTABLE "%s" CACHE PATH "")\n\n' % sphinx_build_exe)
        cfg.close()        
        
        
        