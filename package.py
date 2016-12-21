#!/bin/env python
###############################################################################
#
# TODO: STRAWMAN RELEASE HEADER
# 
###############################################################################


###############################################################################
#
# file: package.py
#
###############################################################################

import subprocess
import sys
import datetime
import os

def create_package(output_file=None):
    repo_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if output_file is None:
        suffix = "tar";
        t = datetime.datetime.now()
        output_file = "%s.%04d.%02d.%02d.%s" % (repo_name,t.year,t.month,t.day,suffix)
    if output_file.endswith(".gz"):
        output_file = output_file[:-3]
    cmd = "git archive --format=tar --prefix=%s/ HEAD > %s; " % (repo_name,output_file)
    cmd += "cd ../; tar -rf %s/%s %s/.git; cd %s; " % (repo_name,output_file,repo_name,repo_name)
    cmd += "gzip %s; " % output_file
    print "[exe: %s]" % cmd
    subprocess.call(cmd,shell=True)
    
if __name__ == "__main__":
    ofile  = None
    if len(sys.argv) > 1:
        ofile  = sys.argv[1]
    create_package(ofile)


