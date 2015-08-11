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
# For details, see: http://scalability-llnl.github.io/conduit/.
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
file: update_lc_conduit_docs.py
description:
 Copies sphinx docs to publish them on LLNL's CZ webserver. 

"""

import os
import sys
import subprocess


def sexe(cmd,echo=True):
    if echo:
        print "[sexe: %s]" % cmd
    subprocess.call(cmd,shell=True)

def update_lc_docs(src_dir):
    src_dir = os.path.abspath(src_dir)
    dest_dir = "/usr/global/web-pages/lc/www/conduit/"
    print "[copied docs from %s to %s]" % (src_dir,dest_dir)
    sexe("cp -R %s/* %s" % (src_dir,dest_dir))
    sexe("chgrp -R bdiv %s" % (dest_dir))
    sexe("chmod -R a+rX %s" % (dest_dir))
    sexe("chmod -R g+rwX %s" % (dest_dir))


if __name__ == "__main__":
    nargs = len(sys.argv)
    modify_files = False
    if nargs < 1:
        print "usage: python update_lc_conduit_docs.py "
        print "[sphinx_html_dir]"
        sys.exit(-1)
    src_dir = sys.argv[1]
    update_lc_docs(src_dir)




