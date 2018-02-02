from spack import *

class PyAlabaster(Package):
    """A configurable sidebar-enabled Sphinx theme"""
    homepage = "https://github.com/bitprophet/alabaster"
    url      = "https://pypi.python.org/packages/source/a/alabaster/alabaster-0.7.7.tar.gz"

    version('0.7.7', '957c665d7126dea8121f98038debcba7')

    extends('python')

    def install(self, spec, prefix):
        # simply install to the spack python
        python('setup.py', 'install') 


