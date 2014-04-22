#!/bin/env python

import subprocess
import sys
import datetime
import os

def create_package(bundle=False):
    t = datetime.datetime.now()
    repo_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    suffix = "tar"
    if bundle:
        suffix = "git.bundle"
    output_file = "%s.%04d.%02d.%02d.%s" % (repo_name,t.year,t.month,t.day,suffix)
    if bundle:
        cmd = "git bundle create %s -all" % output_file
    else: # cyrus' old method
        cmd = "git archive --format=tar --prefix=%s/ HEAD > %s; " % (repo_name,output_file)
        cmd += "cd ../; tar -rf %s/%s %s/.git; cd %s; " % (repo_name,output_file,repo_name,repo_name)
        cmd += "gzip %s; " % output_file
    print "[exe: %s]" % cmd
    subprocess.call(cmd,shell=True)
    
if __name__ == "__main__":
    bundle=True
    if len(sys.argv) > 1:
        bundle = sys.argv[1].lower().startswith("t")
    create_package(bundle)


