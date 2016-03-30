.. ############################################################################
.. # Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
.. # 
.. # Produced at the Lawrence Livermore National Laboratory
.. # 
.. # LLNL-CODE-666778
.. # 
.. # All rights reserved.
.. # 
.. # This file is part of Conduit. 
.. # 
.. # For details, see: http://llnl.github.io/conduit/.
.. # 
.. # Please also read conduit/LICENSE
.. # 
.. # Redistribution and use in source and binary forms, with or without 
.. # modification, are permitted provided that the following conditions are met:
.. # 
.. # * Redistributions of source code must retain the above copyright notice, 
.. #   this list of conditions and the disclaimer below.
.. # 
.. # * Redistributions in binary form must reproduce the above copyright notice,
.. #   this list of conditions and the disclaimer (as noted below) in the
.. #   documentation and/or other materials provided with the distribution.
.. # 
.. # * Neither the name of the LLNS/LLNL nor the names of its contributors may
.. #   be used to endorse or promote products derived from this software without
.. #   specific prior written permission.
.. # 
.. # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
.. # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
.. # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
.. # ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
.. # LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
.. # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
.. # DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
.. # OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
.. # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
.. # STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
.. # IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
.. # POSSIBILITY OF SUCH DAMAGE.
.. # 
.. ############################################################################

===================
mesh
===================


Protocol
----------

The mesh blueprint protocol defines a computational mesh using a Coordinate Set (*coords*), a Topology (*topology*), zero or more Fields (*fields*), and optional State information (*state*).


Coordinates 
~~~~~~~~~~~~~~~~~~~~~

To define a computational mesh, the first required entry is a set of spatial coordinate tuples that can underpin a mesh topology.

Coordinate Systems
==================

The mesh blueprint protocol supports sets of spatial coordinates from the following three coordinate systems:

* Cartesian: {x,y,z}
* Cylindrical: {r,z}
* Spherical: {r,theta,phi}

Coordinate Sets
==================
The mesh blueprint protocol supports three types of Coordinate Sets: *uniform*, *rectilinear*, and *explicit*.  To conform to the protocol, the path **coords** must be an *Object* that contains entries as outlined below: 

* **uniform**

   An implicit coordinate set defined as the cartesian product of i,j,k dimensions starting at an *origin* (ex: {x,y,z}) using a given *spacing* (ex: {dx,dy,dz}).

  * Cartesian
  
  
    * coords/type: “uniform”
    * coords/dims/{i,j,k}
    * coords/origin/{x,y,z} (optional, default = {0,0,0})
    * coords/spacing/{dx,dy,dz} (optional, default = {1,1,1})


  * Cylindrical
  
  
    * coords/type: “uniform”
    * coords/dims/{i,j}
    * coords/origin/{r,z} (optional, default = {0,0})
    * coords/spacing/{dr,dz} (optional, default = {1,1})


  * Spherical
  
  
    * coords/type: “uniform”
    * coords/dims/{i,j}
    * coords/origin/{r,theta,phi} (optional, default = {0,0,0})
    * coords/spacing/{dr,dtheta, dphi} (optional, default = {1,1,1})


* **rectilinear** 

  An implicit coordinate set defined as the cartesian product of passed coordinate arrays.
  
  * Cartesian
  
  
    * coords/type: “rectilinear”
    * coords/values/{x,y,z}

  * Cylindrical:
  
    * coords/type: “rectilinear”
    * coords/values/{r,z}

  * Spherical


    * coords/type: “uniform”
    * coords/values/{r,theta,phi}


* **explicit**

  An explicit set of coordinates, that conforms to the **mcarray** blueprint protocol.

  * Cartesian
  
  
    * coords/type: “explicit”
    * coords/values/{x,y,z}

  * Cylindrical
  
  
    * coords/type: “explicit”
    * coords/values/{r,z}

  * Spherical
  
  
    * coords/type: “explicit”
    * coords/values/{r,theta,phi}


Toplogies
~~~~~~~~~~~~~~~~~~~~~
The next entry required to describe a computational mesh is the topology of the elements of the mesh. To conform to the protocol, the path **topology** must be an *Object* that contains one of the topology descriptions outlined below.

Topology Nomenclature 
************************

(note on {nodes+zones} vs {vertices+cells} vs {points + elements})

The mesh blueprint protocol uses {points,edges,faces,elements}

Element Shape Names
************************

* supported: point, line, tri, quad, tet, hex
* future: polygon, polyhedron

Implicit 
==========

The mesh blueprint protocol accepts three implicit ways to define a grid of elements on top of a coordinate set. For coordinate set with 1D coordinate tuples, *line* elements are used, for sets with 2D coordinate tuples *quad* elements are used, and for 3D coordinate tuples *hex* elements are used.


* **uniform**: An implicit topology that defines a grid of elements on top of a *uniform * coordinate set. 
   
   
   * topology/type: “uniform”
   * topology/elements/origin/{i0,j0,k0} (optional, default = {0,0,0})
   
* **rectilinear**: An implicit topology that defines a grid of elements on top of a *rectilinear* coordinate set. 
   
   * topology/type: “rectilinear”
   * topology/elements/origin/{i0,j0,k0} (optional, default = {0,0,0})

.. attention::
   (can we collapse uniform + rectilinear?)
   
   
   * topology/type: “structured”
   * topology/elements/dims: "implicit"
   * topology/elements/origin/{i0,j0,k0} (optional, default = {0,0,0})


* **structured**: An implicit topology that defines a grid of elements on top of an *explicit* coordinate set.
  

  * topology/type = “structured”
  * topology/elements/dims/{i,j,k}
  * topology/elements/origin/{i0,j0,k0} (optional, default = {0,0,0})


Explicit (Unstructured)
========================

Single Shape Topology
************************

  * topology/type: “unstructured”
  * topology/elements/shape: (shape name)
  * topology/elements/connectivity: (index array)


Mixed Shape Toplogies 
************************
* **list** - A Node in the *List* role, that contains a children that conform to the *Single Shape Topology* case. 

* **object** - A Node in the *Object* role, that contains a children that conform to the *Single Shape Topology* case. 


* **stream** - (strem description)
  (specifying stream ids and stream connectivity)


  * topology/elements/element_types: ()
  * topology/elements/stream: ()

Indexed Streams
^^^^^^^^^^^^^^^^^^^

* Stream of Indexed Elements


    * topology/elements/element_types: ()
    * topology/elements/element_index/stream_ids: ()
    * topology/elements/element_index/offsets: ()
    * topology/elements/stream: ()
    
* Stream of Contiguous Segments of Element Types


    * topology/elements/element_types: ()
    * topology/elements/segment_index/stream_ids: ()
    * topology/elements/segment_index/element_counts: ()
    * topology/elements/stream: ()

Fields
~~~~~~~~~~~~~~~~~~~~~

(fields intro). (path **fields**, is an *Object* with the following entries)

Field 
^^^^^^^^^^^^^^^^^^^
(a field is a **mcarray** with extra metadata about how it is associated with elements of the topology)

State
~~~~~~~~~~~~~~~~~~~~~
(state intro). (path **state**, is an *Object* with the following entries)


Transforms
------------------

Coordinate Sets
~~~~~~~~~~~~~~~~~~~~~

Toplogies
~~~~~~~~~~~~~~~~~~~~~



