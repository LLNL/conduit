from spack import *

class Py3Six(Package):
    """Python 2 and 3 compatibility utilities"""
    homepage = "http://pypi.python.org/pypi/six/"
    url      = "https://pypi.python.org/packages/source/s/six/six-1.9.0.tar.gz"

    version('1.9.0', '476881ef4012262dfc8adc645ee786c4')

    depends_on("py3-tz")

    extends('python3')

    def install(self, spec, prefix):
        # simply install to the spack python
        python('setup.py', 'install') 


