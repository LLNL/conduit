###############################################################################
# Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see https://lc.llnl.gov/conduit/.
# 
# Please also read conduit/LICENSE
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.
# 
###############################################################################

"""
 file: uberenv.py

 description: uses spack to install the external third party libs use by conduit.

"""

import os
import sys
import subprocess
import shutil
import socket
import platform

from os import environ as env

from optparse import OptionParser
from os.path import join as pjoin


def sexe(cmd):
    "Basic shell call helper."
    print "[sexe:%s ]" % cmd
    subprocess.call(cmd,shell=True)

def parse_args():
    "Parses args from command line"
    parser = OptionParser()
    # where to install
    parser.add_option("--prefix",
                      dest="prefix",
                      default="uberenv_libs",
                      help="destination dir")
    # what compiler to use
    parser.add_option("--spec",
                      dest="spec",
                      default=None,
                      help="spack compiler spec")
    # try to use openmp support
    parser.add_option("--python3",
                      dest="python3",
                      action="store_true",
                      help="use python3 instead of python2")
    # try to use openmp support
    parser.add_option("--force",
                      dest="force",
                      default="uberenv-conduit",
                      help="force rebuild of these packages")   
    # parse args
    opts, extras = parser.parse_args()
    # we want a dict b/c the values could 
    # be passed without using optparse
    opts = vars(opts)
    return opts, extras

def uberenv_compilers_yaml_file():
    # path to compilers.yaml, which we will for compiler setup for spack
    compilers_yaml = pjoin(os.path.split(os.path.abspath(__file__))[0],
                           "compilers.yaml")
    if not os.path.isfile(compilers_yaml):
        print "[failed to find uberenv 'compilers.yaml' file]"
        sys.exit(-1)
    return compilers_yaml


def patch_spack(spack_dir,compilers_yaml,pkgs):
    # force uberenv config
    spack_lib_config = pjoin(spack_dir,"lib","spack","spack","config.py")
    src = open(spack_lib_config).read()
    src += "#UBERENV: force only site config"
    src += "config_scopes = config_scopes[0]\n\n"
    # copy in the compiler spec
    print "[copying uberenv compiler specs]"
    spack_etc = pjoin(spack_dir,"etc")
    if not os.path.isdir(spack_etc):
        os.mkdir(spack_etc)
    spack_etc = pjoin(spack_etc,"spack")
    if not os.path.isdir(spack_etc):
        os.mkdir(spack_etc)
    sexe("cp %s spack/etc/spack" % compilers_yaml)
    dest_spack_pkgs = pjoin(spack_dir,"var","spack","packages")
    # hot-copy our packages into spack
    sexe("cp -Rf %s %s" % (pkgs,dest_spack_pkgs))

def main():
    """
    clones and runs spack to setup our third_party libs and
    creates a host-config.cmake file that can be used by strawman.
    """
    # parse args from command line
    opts, extras = parse_args()
    print "[uberenv options: %s]" % str(opts)
    if "darwin" in platform.system().lower():
        dep_tgt = platform.mac_ver()[0]
        dep_tgt = dep_tgt[:dep_tgt.rfind(".")]
        print "[setting MACOSX_DEPLOYMENT_TARGET to %s]" % dep_tgt
        env["MACOSX_DEPLOYMENT_TARGET"] = dep_tgt
    # setup default spec
    if opts["spec"] is None:
        if "darwin" in platform.system().lower():
            opts["spec"] = "%clang"
        else:
            opts["spec"] = "%gcc"
    # get the current working path, and the glob used to identify the 
    # package files we want to hot-copy to spack
    uberenv_path = os.path.split(os.path.abspath(__file__))[0]
    pkgs = pjoin(uberenv_path, "packages","*")
    # setup destination paths
    dest_dir = os.path.abspath(os.path.abspath(opts["prefix"]))
    print dest_dir
    dest_spack = pjoin(dest_dir,"spack")
    dest_spack_pkgs = pjoin(dest_spack,"var","spack","packages")
    # print a warning if the dest path already exists
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    else:
        print "[info: destination '%s' already exists]"  % dest_dir
    if os.path.isdir(dest_spack):
        print "[info: destination '%s' already exists]"  % dest_spack
    if not opts["force"].count("uberenv-conduit") == 1:
         opts["force"] =  opts["force"] + " uberenv-conduit"
    compilers_yaml = uberenv_compilers_yaml_file()
    # clone spack into the dest path
    os.chdir(dest_dir)
    sexe("git clone https://github.com/llnl/spack.git")
    # twist spack's arms 
    patch_spack(dest_spack,compilers_yaml,pkgs)
    # for things we want to force: clean up stages and uninstall them
    force = [ f + opts["spec"] for f in opts["force"].split()]
    force = " ".join(force)
    print "[forcing uninstall of %s]" % force
    sexe("spack/bin/spack clean %s" % force)
    sexe("spack/bin/spack uninstall %s" % force)
    # use the uberenv package to trigger the right builds and build an host-config.cmake file
    sexe("spack/bin/spack install uberenv-conduit " + opts["spec"])


if __name__ == "__main__":
    main()

