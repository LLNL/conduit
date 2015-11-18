from spack import *

class Hdf5(Package):
    """HDF5 is a data model, library, and file format for storing and managing
       data. It supports an unlimited variety of datatypes, and is designed for
       flexible and efficient I/O and for high volume and complex data.
    """

    homepage = "http://www.hdfgroup.org/HDF5/"
    url      = "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.7/src/hdf5-1.8.7.tar.gz"

    version('1.8.7', '37711d4bcb72997e93d495f97c76c33a')

    depends_on("szip")

    def install(self, spec, prefix):
        configure(
            "--prefix=%s" % prefix,
            "--with-szlib=%s" % spec['szip'].prefix,
            "--enable-shared")
        make()
        make("install")
