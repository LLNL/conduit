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

# Specify the sources of the pure python and compiled portions of our module.
SET(blueprint_py_python_sources  py_src/__init__.py)
SET(blueprint_py_cpp_sources     conduit_blueprint_python.cpp)

# Setup our module
PYTHON_ADD_HYBRID_MODULE(conduit_blueprint_python
                         python-modules
                         conduit/blueprint
                         setup.py
                         "${blueprint_py_python_sources}"
                         ${blueprint_py_cpp_sources})

# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_python conduit conduit_blueprint)

#############################################################
# blueprint.mcarray
#############################################################

# add mcarray submodule
PYTHON_ADD_COMPILED_MODULE(conduit_blueprint_mcarray_python
                           # dest_dir 
                           python-modules
                           # python module dir
                           conduit/blueprint/mcarray
                           # c srcs
                           conduit_blueprint_mcarray_python.cpp)


# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_mcarray_python conduit conduit_blueprint)

# add mcarray examples submodule
PYTHON_ADD_COMPILED_MODULE(conduit_blueprint_mcarray_examples_python
                           # dest_dir 
                           python-modules
                           # python module dir
                           conduit/blueprint/mcarray/examples
                           # c srcs
                           conduit_blueprint_mcarray_examples_python.cpp)


# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_mcarray_examples_python conduit conduit_blueprint)

#############################################################
# blueprint.mesh
#############################################################

# add mesh submodule
PYTHON_ADD_COMPILED_MODULE(conduit_blueprint_mesh_python
                           # dest_dir 
                           python-modules
                           # python module dir
                           conduit/blueprint/mesh
                           # c srcs
                           conduit_blueprint_mesh_python.cpp)


# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_mesh_python conduit conduit_blueprint)


# add mesh examples submodule
PYTHON_ADD_COMPILED_MODULE(conduit_blueprint_mesh_examples_python
                           # dest_dir 
                           python-modules
                           # python module dir
                           conduit/blueprint/mesh/examples
                           # c srcs
                           conduit_blueprint_mesh_examples_python.cpp)


# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_mesh_examples_python conduit conduit_blueprint)


