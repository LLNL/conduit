###############################################################################
# Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see: http://software.llnl.gov/conduit/.
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
file: conduit_relay_entangle.py

Python script that helps establish ssh tunneled connections to htpasswd 
secured  web servers running on HPC clusters.
 
Basic Recipe

1. Run the entangle register command on the HPC cluster to create a new 
   key, htpasswd file, and server info json file. 

2. Start the web server on the HPC cluster using the generated htpasswd 
   file and bind to localhost:{port} (port = 9000 or the number provided 
   via the --port option in step 1).

3. Download the entangle server info json file to the client computer via
   a secure method (ssh, scp, lorenz, etc).

4. Run the entangle tunnel command on the client computer with the server 
   info json file. (This uses ssh to establish a local ssh tunnel, via 
   ssh -L)

5. Open localhost:{port} using a web browser on the client machine to 
   access the remote web server.

== Server Usage Details =-

 python conduit_relay_entangle.py --register

 optional arguments:
  
   --port      server port number       (default: 9000)
   --obase     output file base name    (default: entangle-{timestamp})
   --hostname  explicit server hostname (default: not used)
   --gateway   additional ssh gateway   (default: not used)

 The register command generates new one-time key and creates the following
 files:

 {obase}.json 

    A json file with all of the server details (including the one-time key)
    A client uses this file to establish a ssh tunnel with the 
    conduit_relay_entangle.py 
    tunnel command (see: client usage details)

   *NOTE*: Since this file contains the key, only transfer the info in this 
           file via a secure channel (ssh, lorenz, etc).

 {obase}.htpasswd 

   A digest style htpasswd file for use by the web server for authentication.

== Client Usage Details ==

 python conduit_relay_entangle.py --tunnel --server_info {entangle-info.json}

 The tunnel command uses the info from the passed server json file to setup 
 a ssh tunnel. The ssh connection follows normal authentication rules. If a
 gateway is required to reach the server, make sure passwordless ssh is 
 enabled between the gateway and final host. The command also displays the 
 details from the server json file, including the username and password 
 needed to log into the web server.

 After the tunnel is established, you can open localhost:{port} in the web
 browser on the client computer and enter the username and password to
 access the web server.
"""

#-----------------------------------------------------------------------------#
# imports 
#-----------------------------------------------------------------------------#

import datetime
import os
import socket
import subprocess
import random
import hashlib
import base64
import json
import optparse

#-----------------------------------------------------------------------------#
# helper functions
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
def timestamp(t=None,sep="_"):
    """ Creates a timestamp that can easily be included in a file name. """
    if t is None:
        t = datetime.datetime.now()
    sargs = (t.year,t.month,t.day,t.hour,t.minute,t.second)
    sbase = sep.join(["%04d","%02d","%02d","%02d","%02d","%02d"])
    return  sbase % sargs

#-----------------------------------------------------------------------------#
def username():
    """ Return username from env. """
    return os.environ["USER"]

#-----------------------------------------------------------------------------#
def hostname(full=True):
    """ Returns the hostname of the current machine. """
    return socket.gethostbyaddr(socket.gethostname())[0]

#-----------------------------------------------------------------------------#
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


#-----------------------------------------------------------------------------#
# key gen and htpasswd gen functions
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
def generate_key():
    """ Generates string rep of a new random 256-bit key. """
    return hex(random.SystemRandom().getrandbits(256))

#-----------------------------------------------------------------------------#
def generate_htpasswd(key):
    """ Generates a htpasswd string for a given key. """
    digest_input = "{user}:localhost:{key}".format(user=username(),
                                                   key=key)
    md5 = hashlib.md5()
    md5.update(digest_input)
    digest = md5.hexdigest()
    res = "{user}:localhost:{digest}".format(user = username(),
                                             digest = digest)
    return res


#-----------------------------------------------------------------------------#
# main entry points
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
def register(opts):
    """ register command (for servers) """
    # process options
    tstamp = timestamp()
    port  = opts['port']
    
    # check if obase was pased
    # if not, create one using the timestamp
    if opts['obase'] is None:
        opts['obase'] = "entangle-" + tstamp
    obase = opts['obase']

    # check for explicit host name
    if opts["hostname"] is None:
        host = hostname()
    else:
        host = opts["hostname"]

    # prep output file names
    info_fname     = obase + ".json"
    htpasswd_fname = obase + ".htpasswd"

    # generate a new random key
    key = generate_key()

    # construct info dict
    info = {"timestamp": tstamp,
            "port": port,
            "key":  key,
            "host": host,
            "user": username(),
            "htpasswd": os.path.abspath(htpasswd_fname),
            "gateway":  opts["gateway"]}

    # create info json 
    info_json = json.dumps(info,indent=2)
    print("[server info: {fname}]".format(fname=info_fname))
    print(info_json)
    
    # write json info file
    open(info_fname,"w").write(info_json)
    
    # write htpasswd file
    htpasswd_txt = generate_htpasswd(key)
    print("")
    print("[htpasswd: {fname}]".format(fname=htpasswd_fname))
    print(htpasswd_txt)
    print("")
    open(htpasswd_fname,"w").write( htpasswd_txt)

#-----------------------------------------------------------------------------#
def tunnel(opts):
    """ tunnel command (for clients) """
    # process options
    # make sure we have a server info file
    if opts["server_info"] is None:
        print("{ERROR} --tunnel requires --server-info argument "
              "with path to server json info file ")
        return

    # load server info
    sinfo_fname = opts["server_info"]
    info = json.load(open(sinfo_fname))

    # pull what we need for our ssh command
    port = info["port"]
    user = info["user"]
    host = info["host"]
    
    # print info
    info_json = json.dumps(info,indent=2)
    print("[server info: {fname}]".format(fname=sinfo_fname))
    print(info_json)

    # open the tunnel
    cmd = "ssh"
    if info["gateway"] is not None:
        # single hope gateway
        gw = info["gateway"]
        cmd += " -L {port}:localhost:{port} {user}@{host}".format(port = port,
                                                                  user = user,
                                                                  host = gw)
        cmd += " ssh"
    cmd += " -L {port}:localhost:{port} {user}@{host}".format(port = port,
                                                              user = user,
                                                              host = host)
    print("")
    print "[opening ssh tunnel]"
    sexe(cmd,echo=True) 


#-----------------------------------------------------------------------------#
def parse_args():
    """ Parses arguments passed to conduit_relay_entangle.py """
    parser = optparse.OptionParser()
    parser.add_option("-r",
                      "--register",
                      dest="register",
                      default=False,
                      action="store_true",
                      help="generate a new client/server key and save "
                            "htpasswd and info files")
    parser.add_option("-p",
                      "--port",
                      type="int",
                      default=9000,
                      help="register: server port")
    parser.add_option("-o",
                      "--obase",
                      default=None,
                      help="register: output file base name "
                           "[default=entangle-{timestamp}]")
    parser.add_option("-g",
                      "--gateway",
                      default=None,
                      help="register: additional gateway for client ssh "
                            "tunnel")
    parser.add_option("--hostname",
                      default=None,
                      help="register: explicitly set hostname for client")
    parser.add_option("-t",
                      "--tunnel",
                      dest="tunnel",
                      default=False,
                      action="store_true",
                      help="establish a ssh tunnel to a server using an "
                            "entangle info file")
    parser.add_option("-s",
                      "--server-info",
                      default=None,
                      dest="server_info",
                      help="tunnel: json file with entangle server info")

    # parse args
    opts, extra = parser.parse_args()

    # get a dict with the resulting args
    opts = vars(opts)
    
    return opts, parser

#-----------------------------------------------------------------------------#
def main():
    """ Main entry point for conduit_relay_entangle.py """
    opts, parser = parse_args()
    if opts["register"]:
        register(opts)
    elif opts["tunnel"]:
        tunnel(opts)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()






