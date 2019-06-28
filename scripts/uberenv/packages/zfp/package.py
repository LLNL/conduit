# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Zfp(CMakePackage):
    """zfp is an open source C/C++ library for high-fidelity, high-throughput
       lossy compression of floating-point and integer multi-dimensional
       arrays.
    """

    homepage = 'http://computation.llnl.gov/projects/floating-point-compression'
    url      = 'https://github.com/LLNL/zfp/releases/download/0.5.5/zfp-0.5.5.tar.gz'

    version('0.5.5', sha256='fdf7b948bab1f4e5dccfe2c2048fd98c24e417ad8fb8a51ed3463d04147393c5')
    version('0.5.4', sha256='768a05ed9bf10e54ac306f90b81dd17b0e7b13782f01823d7da4394fd2da8adb')

    variant('bsws',
        default='64',
        values=('8', '16', '32', '64'),
        multi=False,
        description='Bit stream word size: use smaller for finer ' \
            'rate granularity. Use 8 for H5Z-ZFP filter.')

    variant('shared', default=True,
            description='Build shared versions of the library')

    def cmake_args(self):
        spec = self.spec

        cmake_args = [ "-DZFP_BIT_STREAM_WORD_SIZE={0}".format(spec.variants['bsws'].value) ]
        if '+shared' in spec:
            cmake_args.append("-DBUILD_SHARED_LIBS=ON")
        else:
            cmake_args.append("-DBUILD_SHARED_LIBS=OFF")
        return cmake_args
