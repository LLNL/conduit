###############################################################################
# Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_relay_python_exports.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h")
                
# Setup our modules
PYTHON_ADD_HYBRID_MODULE(conduit_relay_python
                         # dest dir
                         python-modules
                         # python module dir
                         conduit/relay
                         # python setup
                         setup.py
                         # python srcs
                         py_src/__init__.py
                         py_src/io/__init__.py
                         py_src/io/blueprint/__init__.py
                         py_src/mpi/__init__.py
                         py_src/web/__init__.py
                         # c srcs + headers
                         conduit_relay_python.cpp
                         ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h)

# link with the proper libs
target_link_libraries(conduit_relay_python conduit conduit_relay)

# add relay io submodule
PYTHON_ADD_COMPILED_MODULE(conduit_relay_io_python
                           # dest_dir 
                           python-modules
                           # python module dir
                           conduit/relay/io
                           # c srcs + headers
                           conduit_relay_io_python.cpp
                           ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h)


# link with the proper libs (beyond python)
target_link_libraries(conduit_relay_io_python conduit conduit_relay)

# add relay io blueprint submodule
PYTHON_ADD_COMPILED_MODULE(conduit_relay_io_blueprint_python
                           # dest_dir 
                           python-modules
                           # python module dir
                           conduit/relay/io/blueprint
                           # c srcs + headers
                           conduit_relay_io_blueprint_python.cpp
                           ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h)


# link with the proper libs (beyond python)
target_link_libraries(conduit_relay_io_python conduit conduit_relay)


# add relay web submodule
PYTHON_ADD_COMPILED_MODULE(conduit_relay_web_python
                           # dest 
                           python-modules
                           # python module dir
                           conduit/relay/web
                           # c srcs + headers
                           conduit_relay_web_python.cpp
                           ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h)

# link with the proper libs (beyond python)
target_link_libraries(conduit_relay_web_python conduit conduit_relay)


################################################################
# If we have mpi, add the conduit relay mpi module
################################################################

if(MPI_FOUND)

    # add relay mpi submodule
    PYTHON_ADD_COMPILED_MODULE(conduit_relay_mpi_python
                               # dest 
                               python-modules
                               # python module dir
                               conduit/relay/mpi
                               # c srcs + headers
                               conduit_relay_mpi_python.cpp
                               ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h)

    # link with the proper libs (beyond python)
    target_link_libraries(conduit_relay_mpi_python conduit conduit_relay_mpi)

endif()

