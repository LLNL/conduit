##############################################################################
# Copyright (c) 2013, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
#
# This file is part of Spack.
# Written by Todd Gamblin, tgamblin@llnl.gov, All rights reserved.
# LLNL-CODE-647188
#
# For details, see https://github.com/llnl/spack
# Please also see the LICENSE file for our notice and the LGPL.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License (as published by
# the Free Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
# conditions of the GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
##############################################################################
from spack import *
import os

class Mpich(Package):
    """MPICH is a high performance and widely portable implementation of
       the Message Passing Interface (MPI) standard."""
    homepage = "http://www.mpich.org"
    url      = "http://www.mpich.org/static/downloads/3.0.4/mpich-3.0.4.tar.gz"
    list_url   = "http://www.mpich.org/static/downloads/"
    list_depth = 2

    #
    # Rolled back to mpich 3.0.4 b/c on 7/29/2015 the other
    # mpich source tarballs disappeared from the mpich website 
    #
    #version('3.2',   'f414cfa77099cd1fa1a5ae4e22db508a')
    #version('3.1.4', '2ab544607986486562e076b83937bba2')
    #version('3.1.3', '93cb17f91ac758cbf9174ecb03563778')
    #version('3.1.2', '7fbf4b81dcb74b07ae85939d1ceee7f1')
    #version('3.1.1', '40dc408b1e03cc36d80209baaa2d32b7')
    #version('3.1',   '5643dd176499bfb7d25079aaff25f2ec')
    version('3.0.4', '9c5d5d4fe1e17dd12153f40bc5b6dbc0')

    variant('verbs', default=False, description='Build support for OpenFabrics verbs.')

    provides('mpi@:3.0', when='@3:')
    provides('mpi@:1.3', when='@1:')

    def setup_dependent_environment(self, spack_env, run_env, dependent_spec):
        spack_env.set('MPICH_CC', spack_cc)
        spack_env.set('MPICH_CXX', spack_cxx)
        spack_env.set('MPICH_F77', spack_f77)
        spack_env.set('MPICH_F90', spack_fc)
        spack_env.set('MPICH_FC', spack_fc)

    def setup_dependent_package(self, module, dep_spec):
        self.spec.mpicc  = join_path(self.prefix.bin, 'mpicc')
        self.spec.mpicxx = join_path(self.prefix.bin, 'mpic++')
        self.spec.mpifc  = join_path(self.prefix.bin, 'mpif90')
        self.spec.mpif77 = join_path(self.prefix.bin, 'mpif77')

    def install(self, spec, prefix):
        config_args = ["--prefix=" + prefix,
                       "--enable-shared"]

        # Variants
        if '+verbs' in spec:
            config_args.append("--with-ibverbs")
        else:
            config_args.append("--without-ibverbs")

        # TODO: Spack should make it so that you can't actually find
        # these compilers if they're "disabled" for the current
        # compiler configuration.
        found_fortran = False;
        if self.compiler.f77 and os.path.isfile(self.compiler.f77):
            found_fortran = True;
        else:
            config_args.append("--disable-f77")
            
        if self.compiler.fc and os.path.isfile(self.compiler.fc):
            found_fortran = True;
        else:
            config_args.append("--disable-fc")

        if not found_fortran:
            config_args.append("--disable-fortran")

        configure(*config_args)
        make()
        make("install")

        self.filter_compilers()


    def filter_compilers(self):
        """Run after install to make the MPI compilers use the
           compilers that Spack built the package with.

           If this isn't done, they'll have CC, CXX, F77, and FC set
           to Spack's generic cc, c++, f77, and f90.  We want them to
           be bound to whatever compiler they were built with.
        """
        bin = self.prefix.bin
        mpicc  = os.path.join(bin, 'mpicc')
        mpicxx = os.path.join(bin, 'mpicxx')
        mpif77 = os.path.join(bin, 'mpif77')
        mpif90 = os.path.join(bin, 'mpif90')

        spack_cc  = os.environ['CC']
        spack_cxx = os.environ['CXX']
        spack_f77 = os.environ['F77']
        spack_fc  = os.environ['FC']

        kwargs = { 'ignore_absent' : True, 'backup' : False, 'string' : True }
        filter_file('CC="%s"' % spack_cc , 'CC="%s"'  % self.compiler.cc,  mpicc,  **kwargs)
        filter_file('CXX="%s"'% spack_cxx, 'CXX="%s"' % self.compiler.cxx, mpicxx, **kwargs)
        filter_file('F77="%s"'% spack_f77, 'F77="%s"' % self.compiler.f77, mpif77, **kwargs)
        filter_file('FC="%s"' % spack_fc , 'FC="%s"'  % self.compiler.fc,  mpif90, **kwargs)
