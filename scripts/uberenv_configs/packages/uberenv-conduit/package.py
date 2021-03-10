##############################################################################
# Copyright (c) 2013-2021, Lawrence Livermore National Security, LLC.
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

    # These are default choices for development that differ
    # from spacks default choices
    # (for example, spack wants docs off, python off -- by default)

    # default to building docs when using uberenv
    variant("doc",
            default=True,
            description="Build deps needed to create Conduit's Docs")

    variant("python",
            default=True,
            description="Build deps needed for Conduit python support")

    depends_on("hdf5@1.8~mpi",when="+hdf5+hdf5_compat")

    ###################################
    # build phases used by this package
    ###################################
    phases = ['hostconfig']

    def url_for_version(self, version):
        dummy_tar_path =  os.path.abspath(pjoin(os.path.split(__file__)[0]))
        dummy_tar_path = pjoin(dummy_tar_path,"uberenv-conduit.tar.gz")
        url      = "file://" + dummy_tar_path
        return url
