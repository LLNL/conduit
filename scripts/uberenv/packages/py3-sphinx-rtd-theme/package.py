from spack import *

class Py3SphinxRtdTheme(Package):
    """ReadTheDocs.org theme for Sphinx, 2013 version."""
    homepage = "https://pypi.python.org/pypi/sphinx_rtd_theme"
    url      = "https://pypi.python.org/packages/source/s/sphinx_rtd_theme/sphinx_rtd_theme-0.1.9.tar.gz"

    version('0.1.9', '86a25c8d47147c872e42dc84cc66f97b')

    extends('python3')

    # sphinx-rtd-theme requires sphinx which requires sphinx-rtd-theme
    # remove the recursion with this patch
    patch('remove_requirements.patch', level=0)

    def install(self, spec, prefix):
        # simply install to the spack python
        python('setup.py', 'install') 


