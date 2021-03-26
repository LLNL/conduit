! Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
! Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
! other details. No copyright assignment is required to contribute to Conduit.

!-----------------------------------------------------------------------------
!
! file: conduit_fort_and_py_ex.F90
!
!-----------------------------------------------------------------------------

!-----------------------------------------------------------------------------
! Demos creating and passing Conduit Nodes between Fortran and Python
!-----------------------------------------------------------------------------

PROGRAM main
  use iso_c_binding
  ! use the conduit fortran interface
  use conduit
  ! use our example module
  use conduit_fort_and_py_mod
  implicit none

  type(C_PTR) cnode, cnodepy
  real(4), dimension(4) :: my_data
  integer(4), dimension(2) :: my_shape
  integer i

      ! fill our 32-bit 4 integer array
      do i = 1,4
          my_data(i) = i
      enddo
      ! set our shape
      my_shape(1) = 2
      my_shape(2) = 2
  
      cnode = conduit_node_create()
      call conduit_node_set_path_float32_ptr(cnode,"values",my_data, 4_8)
      call conduit_node_set_path_int32_ptr(cnode,"shape",my_shape, 2_8)

      print*,"Hello from Fortran, here is the Node we created:"
      call conduit_node_print(cnode)

      ! pass this node off to python
      call conduit_fort_to_py(cnode)

      ! lets create a node in python and access it in fortran
      cnodepy = conduit_fort_from_py("my_py_node")

      print*,"Hello from Fortran, here is the Node we got from python:"
      ! print our objc from python
      call conduit_node_print(cnodepy)

      ! we own this one so clean it up.
      call conduit_node_destroy(cnode)

END PROGRAM main
