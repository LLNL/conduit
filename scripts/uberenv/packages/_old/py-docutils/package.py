from spack import *

class PyDocutils(Package):
    """Python Documentation Utilities"""
    homepage = "http://docutils.sourceforge.net/"
    url      = "https://pypi.python.org/packages/source/d/docutils/docutils-0.12.tar.gz"

    version('0.12', '4622263b62c5c771c03502afa3157768')

    extends('python')

    def install(self, spec, prefix):
        # simply install to the spack python
        python('setup.py', 'install') 


