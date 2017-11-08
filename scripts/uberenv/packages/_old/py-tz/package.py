from spack import *

class PyTz(Package):
    """World timezone definitions, modern and historical"""
    homepage = "http://pythonhosted.org/pytz"
    url      = "https://pypi.python.org/packages/source/p/pytz/pytz-2015.7.tar.gz"

    version('2015.7', '252bb731883f37ff9c7f462954e8706d')

    extends('python')

    def install(self, spec, prefix):
        # simply install to the spack python
        python('setup.py', 'install') 


