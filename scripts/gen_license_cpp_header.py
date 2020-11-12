# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
file: update_license_header_txt.py
description:
 Simple python script to help with update license header text in files  
 throughout the source tree.
"""

import os
import sys

pattern = {
# c++ style headers
    "hdr":"""//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
""",
    "st": "// "}


def gen_lic_hpp(lic_file,cpyr_hdr_file,hpp_out):
    lic_txt = open(lic_file).readlines()
    cpry_hdr_txt = open(cpyr_hdr_file).readlines()
    # write the lic prelude, then create var to use in c++
    hpp_f = open(hpp_out,"w")
    for l in cpry_hdr_txt:
        hpp_f.write("%s%s" % (pattern["st"],l))
    hpp_f.write("\n")
    hpp_f.write("#ifndef CONDUIT_LICENSE_TEXT_HPP\n")
    hpp_f.write("#define CONDUIT_LICENSE_TEXT_HPP\n\n")
    hpp_f.write("std::string CONDUIT_LICENSE_TEXT = ")
    for l in lic_txt:
        ltxt = l.strip().replace("\"","\\\"")
        hpp_f.write("\"%s\\n\"\n" % (ltxt))
    hpp_f.write("\"\";")
    hpp_f.write("\n\n")
    hpp_f.write("#endif\n\n")

if __name__ == "__main__":
    nargs = len(sys.argv)
    modify_files = False
    if nargs < 4:
        print "usage: python generate_cpp_license_header.py "
        print "[new lic] [lic header] [output file]"
        sys.exit(-1)
    lic_file = sys.argv[1]
    cpyr_hdr = sys.argv[2]
    hpp_out  = sys.argv[3]
    gen_lic_hpp(lic_file,cpyr_hdr,hpp_out)

