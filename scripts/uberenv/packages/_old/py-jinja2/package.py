from spack import *

class PyJinja2(Package):
    """A small but fast and easy to use stand-alone template engine written in pure python."""
    homepage = "http://jinja.pocoo.org/"
    url      = "https://pypi.python.org/packages/source/J/Jinja2/Jinja2-2.8.tar.gz"

    version('2.8', 'edb51693fe22c53cee5403775c71a99e')

    depends_on("py-markupsafe")
    depends_on("py-babel")

    extends('python')

    def install(self, spec, prefix):
        # simply install to the spack python
        python('setup.py', 'install') 


