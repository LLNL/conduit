.. ############################################################################
.. # Copyright (c) 2014, Lawrence Livermore National Security, LLC.
.. # 
.. # Produced at the Lawrence Livermore National Laboratory
.. # 
.. # LLNL-CODE-666778
.. # 
.. # All rights reserved.
.. # 
.. # This file is part of Conduit. 
.. # 
.. # For details, see https://lc.llnl.gov/conduit/.
.. # 
.. # Please also read conduit/LICENSE
.. # 
.. # Redistribution and use in source and binary forms, with or without 
.. # modification, are permitted provided that the following conditions are met:
.. # 
.. # • Redistributions of source code must retain the above copyright notice, 
.. #   this list of conditions and the disclaimer below.
.. # 
.. # • Redistributions in binary form must reproduce the above copyright notice,
.. #   this list of conditions and the disclaimer (as noted below) in the
.. #   documentation and/or other materials provided with the distribution.
.. # 
.. # • Neither the name of the LLNS/LLNL nor the names of its contributors may
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
.. # 
.. # IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
.. # POSSIBILITY OF SUCH DAMAGE.
.. # 
.. ############################################################################

======================
Basic Concepts
======================

*Node* basics
----------------

The *Node* class is the primary object in conduit.

Think of it as a hierarchical variant object.

.. code:: cpp

    Node n;
    n["my"] = "data";
    n.print(); 

.. parsed-literal::

    
    {
      "my": "data"
    }


The *Node* class supports hierarchical construction

.. code:: cpp

    Node n;
    n["my"] = "data";
    n["a/b/c"] = "d";
    n["a"]["b"]["e"] = 64.0;
    n.print();
    
    std::cout << "total bytes: " << n.total_bytes() << std::endl;


.. parsed-literal::

    
    {
      "my": "data",
      "a": 
      {
        "b": 
        {
          "c": "d",
          "e": 64
        }
      }
    }
    total bytes: 15


Behind the scenes, *Node* instances manage a collection of memory spaces

.. code:: cpp

    Node n;
    n["my"] = "data";
    n["a/b/c"] = "d";
    n["a"]["b"]["e"] = 64.0;
    
    
    Node ninfo;
    n.info(ninfo);
    ninfo.print();

.. parsed-literal::

    
    {
      "mem_spaces": 
      {
        "0x7f8c58c03de0": 
        {
          "path": "my",
          "type": "alloced",
          "bytes": 5
        },
        "0x7f8c58c042b0": 
        {
          "path": "a/b/c",
          "type": "alloced",
          "bytes": 2
        },
        "0x7f8c58c042a0": 
        {
          "path": "a/b/e",
          "type": "alloced",
          "bytes": 8
        }
      },
      "total_bytes": 15,
      "total_bytes_compact": 15,
      "total_bytes_alloced": 15,
      "total_bytes_mmaped": 0
    }


Bitwidth Style Types
--------------------------------

When sharing data in scientific codes, knowing the precision of the underlining types is very important.

Conduit uses well defined bitwidth style types (ala numpy) for leaf values.

.. code:: cpp

    Node n;
    uint32 val = 100;
    n["test"] = val;
    n.print();
    n.print_detailed();

.. parsed-literal::

    
    {
      "test": 100
    }
    
    {
      "test": {"dtype":"uint32", "length": 1, "endianness": "little", "value": 100}
    }


Standard C++ numeric types will be mapped by the compiler to bitwidth style types.

.. code:: cpp

    Node n;
    int val = 100;
    n["test"] = val;
    n.print_detailed();

.. parsed-literal::

    
    {
      "test": {"dtype":"int32", "length": 1, "endianness": "little", "value": 100}
    }


Supported Bitwidth Style Types:
 - signed integers: int8,int16,int32,int64
 - unsigned integers: uint8,uint16,uint32,uint64
 - floating point numbers: float32,float64

Using *Generator* instances to parse JSON schemas
---------------------------------------------------

The *Generator* class is used to parse conduit JSON schemas into a *Node*


.. code:: cpp

    Generator g("{test: {dtype: float64, value: 100.0}}","conduit");
    
    Node n(g);
    std::cout << n["test"].as_float64() <<std::endl;
    n.print();
    n.print_detailed();

.. parsed-literal::

    100
    
    {
      "test": 100
    }
    
    {
      "test": {"dtype":"float64", "length": 1, "endianness": "little", "value": 100}
    }


The *Generator* can also parse pure json. For leaf nodes: wide types such as *int64*, *uint64*, and *float64* are inferred.


.. code:: cpp

    Generator g("{test: 100.0}","json");
    
    Node n(g);
    std::cout << n["test"].as_float64() <<std::endl;
    n.print_detailed();
    n.print();

.. parsed-literal::

    100
    
    {
      "test": {"dtype":"float64", "length": 1, "endianness": "little", "value": 100}
    }
    
    {
      "test": 100
    }


Schemas can easily be bound to in-core data


.. code:: cpp

    float64 vals[2];
    Generator g("{a: {dtype: float64, value: 100.0}, b: {dtype: float64, value: 200.0} }",vals);
    
    Node n(g);
    std::cout << n["a"].as_float64() << " vs " << vals[0] << std::endl; 
    std::cout << n["b"].as_float64() << " vs " << vals[1] << std::endl; 
    
    n.print();
                    
    Node ninfo;
    n.info(ninfo);
    ninfo.print();

.. parsed-literal::

    100 vs 100
    200 vs 200
    
    {
      "a": 100,
      "b": 200
    }
    
    {
      "mem_spaces": 
      {
        "0x7fff55795660": 
        {
          "path": "a",
          "type": "external"
        }
      },
      "total_bytes": 16,
      "total_bytes_compact": 16,
      "total_bytes_alloced": 0,
      "total_bytes_mmaped": 0
    }


Compaction and Serialization
--------------------------------

*Nodes* can be compacted for serialization


.. code:: cpp

    float64 vals[] = { 100.0,-100.0,200.0,-200.0,300.0,-300.0,400.0,-400.0,500.0,-500.0};
    Generator g1("{dtype: float64, length: 5, stride: 16}",vals);
    Generator g2("{dtype: float64, length: 5, stride: 16, offset:8}",vals);
    
    
    Node n1(g1);
    n1.print();
    
    Node n2(g2);
    n2.print();
    
    Node ninfo;
    n1.info(ninfo);
    ninfo.print();
    
    
    Node n1c;
    n1.compact_to(n1c);
    
    n1c.print();
    n1c.schema().print();
    n1c.info(ninfo);
    ninfo.print();
    
    Node n2c;
    n2.compact_to(n2c);
    
    n2c.print();
    n2c.info(ninfo);
    ninfo.print();


.. parsed-literal::

    [100, 200, 300, 400, 500]
    [-100, -200, -300, -400, -500]
    
    {
      "mem_spaces": 
      {
        "0x7fff520e1680": 
        {
          "path": "",
          "type": "external"
        }
      },
      "total_bytes": 80,
      "total_bytes_compact": 40,
      "total_bytes_alloced": 0,
      "total_bytes_mmaped": 0
    }
    [100, 200, 300, 400, 500]
    {"dtype":"float64", "length": 5, "offset": 0, "stride": 8, "element_bytes": 8, "endianness": "little"}
    
    {
      "mem_spaces": 
      {
        "0x7f8f88500f50": 
        {
          "path": "",
          "type": "alloced",
          "bytes": 40
        }
      },
      "total_bytes": 40,
      "total_bytes_compact": 40,
      "total_bytes_alloced": 40,
      "total_bytes_mmaped": 0
    }
    [-100, -200, -300, -400, -500]
    
    {
      "mem_spaces": 
      {
        "0x7f8f885006b0": 
        {
          "path": "",
          "type": "alloced",
          "bytes": 40
        }
      },
      "total_bytes": 40,
      "total_bytes_compact": 40,
      "total_bytes_alloced": 40,
      "total_bytes_mmaped": 0
    }


