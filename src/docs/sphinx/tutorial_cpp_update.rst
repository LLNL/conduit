.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

============================================
Node Update Methods
============================================

The *Node* class provides three **update** methods which allow you to easily copy data or  the description of data from a source node. 

- **Node::update(Node &source)**: 

 This method behaves similar to a python dictionary update. Entires from the source Node are copied into the calling Node, here are more concrete details:

 - **If the source describes an Object**: 
 
   - Update copies the children of the source Node into the calling Node. Normal set semantics apply: if a compatible child with the same name already exists in the calling Node, the data will be copied.  If not, the calling Node will dynamically construct children to hold copies of each child of the source Node. 

 - **If the source describes a List**: 
 
   - Update copies the children of the source Node into the calling Node. Normal set semantics apply: if a compatible child already exists in the same list order in the calling Node, the data will be copied.  If not, the calling Node will dynamically construct children to hold copies of each child of the source Node. 

 - **If the source Node describes a leaf data type**: 

   - Update works exactly like a **set** (not true yet).

- **Node::update_compatible(Node &source)**: 

 This method copies data from the children in the source Node that are compatible with children in the calling node. No changes are made where children are incompatible. 

- **Node::update_external(Node &source)**: 

 This method creates children in the calling Node that externally describe the children in the source node. It differs from **Node::set_external(Node &source)** in that **set_external()** will clear the calling Node so it exactly match an external description of the source Node, whereas **update_external()** will only change the children in the calling Node that correspond to children in the source Node.



