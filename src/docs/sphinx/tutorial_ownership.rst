.. #############################################################################
.. # Copyright (c) 2014, Lawrence Livermore National Security, LLC
.. # Produced at the Lawrence Livermore National Laboratory. 
.. # 
.. # All rights reserved.
.. # 
.. # This source code cannot be distributed without further review from 
.. # Lawrence Livermore National Laboratory.
.. #############################################################################

============================================
Memory Ownership Semantics
============================================

*set* vs *set_external* 
--------------------------------

The *Node* class provides two classes of functions that enable passing of data to a *Node*:

- **set**: Makes a copy of the data passed into the *Node*. This may also trigger an allocation if the current data type of the *Node* is incompatible with what was passed. The *Node* assignment operators use their respective *set* variants, so they follow the same copy semantics. 

- **set_external**: Sets up the *Node* to point to the data externally. 

.. code:: cpp

    index_t vsize = 5;
    std::vector<float64> vals(vsize,0.0);
    for(index_t i=0;i<vsize;i++)
    {
        vals[i] = 3.1415 * i;
    }
    
    Node n;
    n["v_owned"] = vals;
    n["v_external"].set_external(vals);
    
    n.info().print(); 
    
    n.print();
    
    vals[1] = -1 * vals[1];
    n.print();


.. parsed-literal::

    {
      "mem_spaces": 
      {
        "0x7fd3d2500240": 
        {
          "path": "v_owned",
          "type": "alloced",
          "bytes": 40
        },
        "0x7fd3d2500000": 
        {
          "path": "v_external",
          "type": "external"
        }
      },
      "total_bytes": 80,
      "total_bytes_compact": 80,
      "total_bytes_alloced": 40,
      "total_bytes_mmaped": 0
    }
    
    {
      "v_owned": [0, 3.1415, 6.283, 9.4245, 12.566],
      "v_external": [0, 3.1415, 6.283, 9.4245, 12.566]
    }
    
    {
      "v_owned": [0, 3.1415, 6.283, 9.4245, 12.566],
      "v_external": [0, -3.1415, 6.283, 9.4245, 12.566]
    }


