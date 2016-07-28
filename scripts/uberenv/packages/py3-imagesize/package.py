from spack import *

class Py3Imagesize(Package):
    """Python pacage for getting image size from png/jpeg/jpeg2000/gif files"""
    homepage = "https://github.com/shibukawa/imagesize_py"
    url      = "https://pypi.python.org/packages/53/72/6c6f1e787d9cab2cc733cf042f125abec07209a58308831c9f292504e826/imagesize-0.7.1.tar.gz"

    version('0.7.1', '976148283286a6ba5f69b0f81aef8052')

    extends('python3')

    def install(self, spec, prefix):
        # simply install to the spack python
        python('setup.py', 'install') 


