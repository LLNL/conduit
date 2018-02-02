from spack import *

class Silo(Package):
    """A Mesh and Field I/O Library and Scientific Database"""
    homepage = "https://wci.llnl.gov/simulation/computer-codes/silo/"
    url      = "http://50.56.174.237/svn/visit/trunk/third_party/silo-4.10.1.tar.gz"

    version('4.10.1', '29e6cdf25e98aef96e1f541167839c6f')

    depends_on("szip")
    depends_on("hdf5")

    def install(self, spec, prefix):
        hdf5  = spec['hdf5']
        szlib = spec["szip"]

        configure( "--prefix=%s" % prefix,
                   "--disable-pythonmodule",
                   "--disable-browser",
                   "--enable-shared",
                   "--disable-silex",
                   "--disable-fortran",
                   "--without-zlib",
                   "--with-szlib=%s" % szlib.prefix,
                   "--with-hdf5=%s,%s" % (hdf5.prefix.include, hdf5.prefix.lib) )
        make()
        make("install")
