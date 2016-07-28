from spack import *

class Py3Snowballstemmer(Package):
    """This package provides 16 stemmer algorithms (15 + Poerter English stemmer) generated from Snowball algorithms."""
    homepage = "https://github.com/shibukawa/snowball_py"
    url      = "https://pypi.python.org/packages/source/s/snowballstemmer/snowballstemmer-1.2.1.tar.gz"

    version('1.2.1', '643b019667a708a922172e33a99bf2fa')

    extends('python3')

    def install(self, spec, prefix):
        # simply install to the spack python
        python('setup.py', 'install') 


