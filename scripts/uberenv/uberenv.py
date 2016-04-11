###############################################################################
# Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

 description: uses spack to install the external third party libs used by a project.

"""

import os
import sys
import subprocess
import shutil
import socket
import platform
import json

from optparse import OptionParser

from os import environ as env
from os.path import join as pjoin


def sexe(cmd,ret_output=False,echo = False):
    """ Helper for executing shell commands. """
    if echo:
        print "[exe: %s]" % cmd
    if ret_output:
        p = subprocess.Popen(cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        res =p.communicate()[0]
        return p.returncode,res
    else:
        return subprocess.call(cmd,shell=True)


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
    # force rebuild of these packages
    parser.add_option("--force",
                      dest="force",
                      default=False,
                      action='store_true',
                      help="force rebuild of uberenv packages")
    # a file that holds settings for a specific project 
    # using uberenv.py 
    parser.add_option("--package_json",
                      dest="package_json",
                      default=pjoin(uberenv_script_dir(),"uberenv_package.json"),
                      help="uberenv package info json file")

    # parse args
    opts, extras = parser.parse_args()
    # we want a dict b/c the values could 
    # be passed without using optparse
    opts = vars(opts)
    return opts, extras

def uberenv_script_dir():
    # returns the directory of the uberenv.py script
    return os.path.dirname(os.path.abspath(__file__))

def uberenv_package_info(package_json):
    # reads package info from json file
    return json.load(open(package_json))

def spack_package_is_installed(pkg,spec):
    # TODO: We need a better way to check this
    # the term colors are undermining me 
    # ( I was trying to check for z installed pacakges, with z > 0
    rcode, output = sexe("spack/bin/spack find " + pkg + spec,
                         ret_output=True,echo=True)
    lines = output.split("\n")
    return len(lines) > 1

def spack_package_deps(pkg,spec):
    # TODO: In the future, we will can use uninstall by hash
    # and we won't need this
    rcode, output = sexe("spack/bin/spack find --deps %s %s" % (pkg,spec),
                         ret_output=True,
                         echo=True)
    lines = [ l.strip() for l in output.split("\n") if l.strip() != ""] 
    lines = [ l[1:] for l in lines if l.startswith("^")]
    # remove term colors that spack uses
    lines = [ l.replace("\x1b[0;36m","") for l in lines]
    lines = [ l.replace("\x1b[0;94m","") for l in lines]
    lines = [ l.replace("\x1b[0m","") for l in lines]
    pkgs = set(lines)
    res = [ pkg + spec for pkg in pkgs]
    return res    

def spack_uninstall_and_clean(pkg):
    print "[forcing uninstall of %s]" % pkg
    sexe("spack/bin/spack uninstall -f %s" % pkg,echo=True)
    sexe("spack/bin/spack clean %s" % pkg,echo=True)

def uberenv_compilers_yaml_file():
    # path to compilers.yaml, which we will for compiler setup for spack
    compilers_yaml = pjoin(uberenv_script_dir(),
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
    dest_spack_pkgs = pjoin(spack_dir,"var","spack","repos","builtin","packages")
    # hot-copy our packages into spack
    sexe("cp -Rf %s %s" % (pkgs,dest_spack_pkgs))


def main():
    """
    clones and runs spack to setup our third_party libs and
    creates a host-config.cmake file that can be used by 
    our project.
    """ 
    # parse args from command line
    opts, extras = parse_args()
    
    pkg_info = uberenv_package_info(opts["package_json"])
    print pkg_info
    uberenv_pkg_name = pkg_info["uberenv_package_name"]
    
    # setup osx deployment target
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
    dest_dir = os.path.abspath(opts["prefix"])
    dest_spack = pjoin(dest_dir,"spack")
    # print a warning if the dest path already exists
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    else:
        print "[info: destination '%s' already exists]"  % dest_dir
    if os.path.isdir(dest_spack):
        print "[info: destination '%s' already exists]"  % dest_spack
    compilers_yaml = uberenv_compilers_yaml_file()
    if not os.path.isdir("spack"):
        print "[info: cloning spack from github]"
        os.chdir(dest_dir)
        # clone spack into the dest path
        sexe("git clone -b develop https://github.com/llnl/spack.git")
    else:
        print "[info: updating spack from github]"
        # if we already have a checkout, clean and update it
        os.chdir(pjoin(dest_dir,"spack"))
        sexe("git clean -f")
        sexe("git pull origin develop")
    os.chdir(dest_dir)
    # twist spack's arms 
    patch_spack(dest_spack,compilers_yaml,pkgs)
    if opts["force"]:
        deps = spack_package_deps(uberenv_pkg_name,opts["spec"])
        for dep in deps:
            spack_uninstall_and_clean(dep)
    if spack_package_is_installed(uberenv_pkg_name,opts["spec"]):
        spack_uninstall_and_clean(uberenv_pkg_name + opts["spec"])
    # use the uberenv package to trigger the right builds and build an host-config.cmake file
    sexe("spack/bin/spack install " + uberenv_pkg_name + opts["spec"],echo=True)


if __name__ == "__main__":
    main()


