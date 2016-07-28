from spack import *

class PyPygments(Package):
    """Pygments is a syntax highlighting package written in Python."""
    homepage = "http://pygments.org/"
    url      = "https://pypi.python.org/packages/source/P/Pygments/Pygments-2.1.tar.gz"

    version('2.1', '84533d22f72de894f6d3907c3ca9eddf')

    extends('python')

    def install(self, spec, prefix):
        # simply install to the spack python
        python('setup.py', 'install') 


