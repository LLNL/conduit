##############################################################################
# Copyright (c) 2013-2017, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
#
# This file is part of Spack.
# Created by Todd Gamblin, tgamblin@llnl.gov, All rights reserved.
# LLNL-CODE-647188
#
# For details, see https://github.com/llnl/spack
# Please also see the NOTICE and LICENSE files for our notice and the LGPL.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License (as
# published by the Free Software Foundation) version 2.1, February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
# conditions of the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
##############################################################################
from spack import *

import socket
import os

from os.path import join as pjoin
from os import environ as env

from conduit import Conduit

class UberenvConduit(Conduit):
    """Conduit is an open source project from Lawrence Livermore National
    Laboratory that provides an intuitive model for describing hierarchical
    scientific data in C++, C, Fortran, and Python. It is used for data
    coupling between packages in-core, serialization, and I/O tasks."""

    version('0.0.0', '8d378ef62dedc2df5db447b029b71200',preferred=True)

    # default to building docs when using uberenv
    variant("doc",
            default=True,
            description="Build deps needed to create Conduit's Docs")

    # stick with cmake 3.8 or 3.9 until we use MPIEXEC_EXECUTABLE for 3.10+
    # in upstream spack package
    depends_on("cmake@3.8.2:3.9.999", when="+cmake")

    def url_for_version(self, version):
        dummy_tar_path =  os.path.abspath(pjoin(os.path.split(__file__)[0]))
        dummy_tar_path = pjoin(dummy_tar_path,"uberenv-conduit.tar.gz")
        url      = "file://" + dummy_tar_path
        return url

    def install(self, spec, prefix):
        """
        Build and install Conduit.
        """
        with working_dir('spack-build', create=True):
            host_cfg_fname = self.create_host_config(spec, prefix)
            # place a copy in the spack install dir for the uberenv-conduit package 
            mkdirp(prefix)
            install(host_cfg_fname,prefix)
            install(host_cfg_fname,env["SPACK_DEBUG_LOG_DIR"])
