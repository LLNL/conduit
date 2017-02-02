##############################################################################
# Copyright (c) 2013-2016, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
#
# This file is part of Spack.
# Created by Todd Gamblin, tgamblin@llnl.gov, All rights reserved.
# LLNL-CODE-647188
#
# For details, see https://github.com/llnl/spack
# Please also see the LICENSE file for our notice and the LGPL.
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
import llnl.util.tty as tty

from spack import *


class Openssl(Package):
    """The OpenSSL Project is a collaborative effort to develop a
       robust, commercial-grade, full-featured, and Open Source
       toolkit implementing the Secure Sockets Layer (SSL v2/v3) and
       Transport Layer Security (TLS v1) protocols as well as a
       full-strength general purpose cryptography library."""
    homepage = "http://www.openssl.org"
    url = "ftp://openssl.org/source/old/1.0.2/openssl-1.0.2j.tar.gz"

    version('1.0.2j', '96322138f0b69e61b7212bc53d5e912b')
    version('1.0.2i', '678374e63f8df456a697d3e5e5a931fb')
    version('1.0.2h', '9392e65072ce4b614c1392eefc1f23d0')
    version('1.0.2g', 'f3c710c045cdee5fd114feb69feba7aa')
    version('1.0.2f', 'b3bf73f507172be9292ea2a8c28b659d')
    version('1.0.2e', '5262bfa25b60ed9de9f28d5d52d77fc5')
    version('1.0.2d', '38dd619b2e77cbac69b99f52a053d25a')
    version('1.0.1u', '130bb19745db2a5a09f22ccbbf7e69d0')
    version('1.0.1t', '9837746fcf8a6727d46d22ca35953da1')
    version('1.0.1r', '1abd905e079542ccae948af37e393d28')
    version('1.0.1h', '8d6d684a9430d5cc98a62a5d8fbda8cf')

    depends_on("zlib")
    parallel = False

    def handle_fetch_error(self, error):
        tty.warn("Fetching OpenSSL failed. This may indicate that OpenSSL has "
                 "been updated, and the version in your instance of Spack is "
                 "insecure. Consider updating to the latest OpenSSL version.")

    def install(self, spec, prefix):
        # OpenSSL uses a variable APPS in its Makefile. If it happens to be set
        # in the environment, then this will override what is set in the
        # Makefile, leading to build errors.
        env.pop('APPS', None)

        env['KERNEL_BITS'] = '64'

        options = ['zlib', 'no-krb5', 'shared']

        config = Executable('./config')
        config('--prefix=%s' % prefix,
               '--openssldir=%s' % join_path(prefix, 'etc', 'openssl'),
               *options)

        # Remove non-standard compiler options if present. These options are
        # present e.g. on Darwin. They are non-standard, i.e. most compilers
        # (e.g. gcc) will not accept them.
        filter_file(r'-arch x86_64', '', 'Makefile')

        make()
        make('install')