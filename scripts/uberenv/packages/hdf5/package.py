from spack import *

class Hdf5(Package):
    """HDF5 is a data model, library, and file format for storing and managing
       data. It supports an unlimited variety of datatypes, and is designed for
       flexible and efficient I/O and for high volume and complex data.
    """

    homepage = "http://www.hdfgroup.org/HDF5/"
    url      = "http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.16.tar.bz2"

    version('1.8.16', '79c1593573ebddf734eee8d43ecfe483')

    depends_on("szip")

    def install(self, spec, prefix):
        configure(
            "--prefix=%s" % prefix,
            "--with-szlib=%s" % spec['szip'].prefix,
            "--enable-shared")
        make()
        make("install")
