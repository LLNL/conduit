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

from .conduit import Conduit

class UberenvConduit(Conduit):
    """Conduit is an open source project from Lawrence Livermore National
    Laboratory that provides an intuitive model for describing hierarchical
    scientific data in C++, C, Fortran, and Python. It is used for data
    coupling between packages in-core, serialization, and I/O tasks."""

    version('0.0.0', 'c8b277080a00041cfc4f64619e31f6d6',preferred=True)
    # default to building docs when using uberenv
    variant("doc",
            default=True,
            description="Build deps needed to create Conduit's Docs")
    variant("adios", default=False, description="Build Conduit ADIOS support")


    # stick with cmake 3.8 or 3.9 until we use MPIEXEC_EXECUTABLE for 3.10+
    # in upstream spack package
    depends_on("cmake@3.8.2:3.9.999", when="+cmake")

    # Try some basic ADIOS configurations. NOTE: these are more extensively
    # covered in the Conduit Spack base class. These seem necessary here too.
    # note: Conduit always depends on hdf5 by default. We have a problem with
    # HDF5. The serial parts of Conduit *cannot* use a parallel version. We
    # must therefore build ADIOS without HDF5 since it cannot use a serial
    # version. This should make it possible for Conduit+ADIOS to have
    # serial HDF5 for relay while still having a parallel ADIOS for relay::mpi::io.
    depends_on("adios+mpi~hdf5", when="+adios+mpi")
    depends_on("adios~mpi~hdf5", when="+adios~mpi")

    # build phases used by this package
    phases = ["configure"]

    def cmake_args(self):
        args = super(UberenvConduit, self).cmake_args()
        return []

    def url_for_version(self, version):
        dummy_tar_path =  os.path.abspath(pjoin(os.path.split(__file__)[0]))
        dummy_tar_path = pjoin(dummy_tar_path,"uberenv-conduit.tar.gz")
        url      = "file://" + dummy_tar_path
        return url

    def configure(self, spec, prefix):
        """
        Create a host config for use in conduit
        """
        print("UberenvConduit.configure")
        with working_dir('spack-build', create=True):
            host_cfg_fname = self.create_host_config(spec, prefix)
            # place a copy in the spack install dir for the uberenv-conduit package 
            mkdirp(prefix)
            install(host_cfg_fname,prefix)
            install(host_cfg_fname,env["SPACK_DEBUG_LOG_DIR"])

