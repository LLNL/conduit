from spack import *

class Py3Babel(Package):
    """Internationalization utilities"""
    homepage = "http://babel.pocoo.org/"
    url      = "https://pypi.python.org/packages/source/B/Babel/Babel-1.3.tar.gz"

    version('1.3', '5264ceb02717843cbc9ffce8e6e06bdb')

    extends('python3')

    depends_on("py3-setuptools")
    depends_on("py3-tz")


    def install(self, spec, prefix):
        # simply install to the spack python
        python('setup.py', 'install') 


