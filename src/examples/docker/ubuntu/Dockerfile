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

FROM ubuntu:latest
MAINTAINER Cyrus Harrison <cyrush@llnl.gov>

# fetch build env
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    gcc \
    g++ \
    gfortran \
    zlib1g-dev \
    python \
    utils \
 && rm -rf /var/lib/apt/lists/*


# untar the current source  (created as part of example_build.sh)
COPY conduit.docker.src.tar /
RUN tar -xf conduit.docker.src.tar

# if you would like to clone conduit master directly, you can use:
#RUN git clone --depth 1 https://github.com/LLNL/conduit.git

# at some sites ssl certs are intercepted, which cases issues fetching 
# tpl sources via https

# to resolve this, either:
# 1) pass the "-k" option to uberenv (recommended), 
# 2) install the proper certs into the image, or
# 3) use  the following commands to disable ssl for git and 
#    curl (both are used by spack):
#RUN git config --global http.sslVerify false
#RUN echo insecure >> ~/.curlrc

# bootstrap third party libs using spack and uberenv
RUN cd conduit && python scripts/uberenv/uberenv.py -k --spec %gcc+mpi~doc~silo

# configure a debug build with cmake
RUN cd conduit && mkdir build-debug
RUN cd conduit/build-debug && \
    ../uberenv_libs/spack/opt/spack/*/*/cmake*/bin/cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=/conduit/install-debug \
    -C ../uberenv_libs/*.cmake \
    ../src

# build, test, and install conduit
RUN cd conduit/build-debug && make
RUN cd conduit/build-debug && make test
RUN cd conduit/build-debug && make install

# open port 9000, for use by conduit relay
EXPOSE 9000

CMD ["/bin/bash"]
