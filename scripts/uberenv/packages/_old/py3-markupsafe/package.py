from spack import *

class Py3Markupsafe(Package):
    """Implements a XML/HTML/XHTML Markup safe string for Python"""
    homepage = "http://github.com/mitsuhiko/markupsafe"
    url      = "https://pypi.python.org/packages/source/M/MarkupSafe/MarkupSafe-0.23.tar.gz"

    version('0.23', 'f5ab3deee4c37cd6a922fb81e730da6e')

    extends('python3')

    def install(self, spec, prefix):
        # simply install to the spack python
        python('setup.py', 'install') 


