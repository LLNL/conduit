from spack import *

class Hdf5(Package):
    """HDF5 is a data model, library, and file format for storing and managing
       data. It supports an unlimited variety of datatypes, and is designed for
       flexible and efficient I/O and for high volume and complex data.
    """

    homepage = "http://www.hdfgroup.org/HDF5/"
    url      = "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.16/src/hdf5-1.8.16.tar.gz"

    version('1.8.17', '7d572f8f3b798a628b8245af0391a0ca',preferred=True)
    version('1.8.16', 'b8ed9a36ae142317f88b0c7ef4b9c618')



    depends_on("szip")

    def install(self, spec, prefix):
        configure(
            "--prefix=%s" % prefix,
            "--with-szlib=%s" % spec['szip'].prefix,
            "--enable-shared")
        make()
        make("install")
