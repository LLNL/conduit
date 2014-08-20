import os
import json

from os.path import join as pjoin
import IPython.core.magic as ipym


@ipym.magics_class
class ConduitCppMagics(ipym.Magics):
    @ipym.cell_magic
    def conduitcpp(self, line, cell=None):
        """Compile, execute C++ code, and return the standard output."""
        build_debug   = os.path.abspath("../../../build-debug/")
        install_debug = os.path.abspath("../../../install-debug/")
        lines = open(pjoin(build_debug,"CMakeCache.txt")).readlines()
        compile_flags = "-I {0:s} -L {1:s} -lconduit".format(pjoin(install_debug,"include/"),
                                                             pjoin(install_debug,"lib/"))
        cxx = [ l.strip().split("=")[-1] for l in lines if l.strip().startswith("CMAKE_CXX_COMPILER:")]
        cxx = cxx[0]
        source_filename = 'temp.cpp'
        program_filename = 'temp.exe'
        # Write the code contained in the cell to the C++ file.
        with open(source_filename, 'w') as f:
            f.write("#include <conduit.h>\n")
            f.write("using namespace conduit;\n")
            f.write("int main(int argc, char **argv)\n{\n")
            f.write(cell)
            f.write("\n}\n")
        # Compile the C++ code into an executable.
        cmd ="{0:s} {1:s} -o {2:s} {3:s} ".format(cxx,
                                                 source_filename, 
                                                 program_filename,
                                                 compile_flags)
        if line.count("echo"):
            print "[compiler cmd: {0:s}]".format(cmd)
        compile = self.shell.getoutput(cmd)
        # Execute the executable and return the output.
        if line.count("echo") and len(compile) > 0:
            print "[compiler output: {0:s}]".format(compile)

        os.environ["LD_LIBRARY_PATH"] = pjoin(install_debug,"lib/");        
        os.environ["DYLD_LIBRARY_PATH"] = pjoin(install_debug,"lib/");
        output = self.shell.getoutput(os.path.abspath(program_filename))
        for l in output:
            print l
        return

def load_ipython_extension(ipython):
    ipython.register_magics(ConduitCppMagics)