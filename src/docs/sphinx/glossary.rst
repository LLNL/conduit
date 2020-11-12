.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

=====================
Glossary
=====================

This page aims to provide succinct descriptions of important concepts in Conduit. 


children
~~~~~~~~~
Used for Node instances in the *Object* and *List* role interfaces. A Node may hold a set of indexed children (List role), or indexed and named children (Object role). In both of these cases the children of the Node can be accessed, or removed via their index. Methods related to this concept include:

 - Node::number_of_children()
 - Node::child(index_t)
 - Node::child_ptr(index_t)
 - Node::operator=(index_t)
 - Node::remove(index_t)

 - Schema::number_of_children()
 - Schema::child(index_t)
 - Schema::child_ptr(index_t)
 - Schema::operator=(index_t)
 - Schema::remove(index_t)

paths
~~~~~~~~~
Used for Node instances in *Object* role interface. In the Object role, a Node has a collection of indexed and named children. Access by name is done via a *path*. The path is a forward-slash separated URI, where each segment maps to Node in a hierarchal tree. Methods related to this concept include:

 - Node::fetch(string)
 - Node::fetch_ptr(string)
 - Node::operator=(string)
 - Node::has_path(string)
 - Node::remove(string)

 - Schema::fetch(string)
 - Schema::fetch_existing(string)
 - Schema::fetch_ptr(string)
 - Schema::operator=(string)
 - Schema::has_path(string)
 - Schema::remove(string)

external
~~~~~~~~~
Concept used throughout the Conduit API to specify ownership for passed data.
When using Node constructors, Generators, or Node::set calls, you have the option of using an external variant. When external is specified, a Node does not own (allocate or deallocate) the memory for the data it holds.


