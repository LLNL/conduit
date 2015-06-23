from spack import *

class Szip(Package):
    """Szip Compression Library"""
    homepage = "https://www.hdfgroup.org/doc_resource/SZIP/"
    url      = "http://50.56.174.237/svn/visit/trunk/third_party/szip-2.1.tar.gz"

    version('2.1', '9cc9125a58b905a4148e4e2fda3fabc6')

    def install(self, spec, prefix):
        configure("--prefix=%s" % prefix)
        make()
        make("install")
