.. #############################################################################
.. # Copyright (c) 2014, Lawrence Livermore National Security, LLC
.. # Produced at the Lawrence Livermore National Laboratory. 
.. # 
.. # All rights reserved.
.. # 
.. # This source code cannot be distributed without further review from 
.. # Lawrence Livermore National Laboratory.
.. #############################################################################

=====================
Glossary
=====================

This page aims to provide succinct description of important concepts in Conduit. 


children
~~~~~~~~~
Used for Node instances in the *Object* and *List* role interfaces. A Node may hold a set of indexed children (List role), or indexed and named children (Object role). In both of these cases the children of the Node can be accessed, or removed via their index. Methods related to this concept include:

 - Node::number_of_children()
 - Node::child(index_t)
 - Node::child_pointer(index_t)
 - Node::operator=(index_t)
 - Node::remove(index_t)

 - Schema::number_of_children()
 - Schema::child(index_t)
 - Schema::child_pointer(index_t)
 - Schema::operator=(index_t)
 - Schema::remove(index_t)

paths
~~~~~~~~~
Used for Node instances in *Object* role interface. In the Object role, a Node has a collection of indexed and named children. Access by name is done via a path. The path is a foward-slash separated URI, where each segment mapps to Node in a hierarchal tree. Methods related to this concept include:

 - Node::fetch(string)
 - Node::fetch_pointer(string)
 - Node::operator=(string)
 - Node::has_path(string)
 - Node::paths(vector of strings)
 - Node::remove(string)

 - Schema::fetch(string)
 - Schema::child(string)
 - Schema::fetch_pointer(string)
 - Schema::operator=(string)
 - Schema::has_path(string)
 - Schema::paths(vector of strings)
 - Schema::remove(string)

external
~~~~~~~~~
Concept used throughout the Conduit API to specify ownership for passed data.
When using Node constructors, Generators, or Node::set calls, you have the option of using an external variant. When external is specified, a Node does not own (or allocate) the memory for the data it holds.


