from spack import *

class PySetuptools(Package):
    """Easily download, build, install, upgrade, and uninstall Python packages."""
    homepage = "https://pypi.python.org/pypi/setuptools"
    url      = "https://pypi.python.org/packages/source/s/setuptools/setuptools-11.3.tar.gz"

    version('11.3.1', '01f69212e019a2420c1693fb43593930')
    version('16.0', '0ace0b96233516fc5f7c857d086aa3ad')
    version('18.1', 'f72e87f34fbf07f299f6cb46256a0b06')

    extends('python')

    def install(self, spec, prefix):
        # python ctypes module doesn't build correctly
        # with icc or clang, it is used by setuptools for
        # windows support, which we don't need.
        # remove use of windows_support in dist.py to avoid the issue
        dist_src = open("setuptools/dist.py").read()
        dist_src = dist_src.replace("from setuptools import windows_support",
                                    "#from setuptools import windows_support")
        dist_src = dist_src.replace("windows_support.hide_file(egg_cache_dir)",
                                    "#windows_support.hide_file(egg_cache_dir)t")
        open("setuptools/dist.py","w").write(dist_src)
        # install into our python
        python('setup.py', 'install')

