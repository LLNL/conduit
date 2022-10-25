.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

.. role:: bash(code)
   :language: bash

Source Code Repo Layout
------------------------
* **src/libs/**

 * **conduit/** - Main Conduit library source
 * **relay/** - Relay libraries source
 * **blueprint/** - Blueprint library source

* **src/tests/**

 * **conduit/** - Unit tests for the main Conduit library
 * **relay/** - Unit tests for Conduit Relay libraries
 * **blueprint/** - Unit tests for Blueprint library
 * **thirdparty/** - Unit tests for third party libraries

* **src/examples/** - Basic examples related to building and using Conduit 

.. (see :ref:`_using_in_another_project` ?)

* **src/docs/** -  Documentation 
* **src/thirdparty_builtin/** - Third party libraries we build and manage directly

