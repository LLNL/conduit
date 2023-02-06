.. # Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Conduit.

.. role:: bash(code)
   :language: bash

Git Development Workflow 
------------------------

Conduit's primary source repository and issue tracker are hosted on github:

https://github.com/llnl/conduit


We are using a **Github Flow** model, which is a simpler variant of the confusingly similar sounding **Git Flow** model.

Here are the basics: 

- Development is done on topic branches off the develop.

- Merge to develop is only done via a pull request.

- The develop should always compile and pass all tests.

- Releases are tagged off of develop.

More details on GitHub Flow:

https://guides.github.com/introduction/flow/index.html

Here are some other rules to abide by:

- If you have write permissions for the Conduit repo, you *can* merge your own pull requests.

- After completing all intended work on branch, please delete the remote branch after merging to develop. (Github has an option to do this after you merge a pull request.)







