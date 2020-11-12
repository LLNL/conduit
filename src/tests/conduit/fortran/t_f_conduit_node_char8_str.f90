!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.

!------------------------------------------------------------------------------
!
! f_conduit_node_float64.f
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module f_conduit_node_char8_str
!------------------------------------------------------------------------------

  use iso_c_binding
  use fruit
  use conduit
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------    
! char8_str tests
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_node_set_char8_str
        type(C_PTR) cnode
        character, pointer :: f_arr(:)
    
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_char8_str")
        !----------------------------------------------------------------------

        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        cnode = conduit_node_create()
        ! n = "howdy");
        call conduit_node_set_char8_str(cnode,"howdy")
        ! n.print();
        call conduit_node_print(cnode)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)

        call conduit_node_destroy(cnode)
    
    end subroutine t_node_set_char8_str
    
    !--------------------------------------------------------------------------
    subroutine t_node_as_char8_str
        type(C_PTR) cnode
        character, pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_char8_str")
        !----------------------------------------------------------------------

        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        cnode = conduit_node_create()
        ! n = "howdy");
        call conduit_node_set_char8_str(cnode,"howdy")
        ! n.print();
        call conduit_node_print(cnode)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! char * res = n.as_char8_str();
        call conduit_node_as_char8_str(cnode,f_arr)
        print *, f_arr
        !
        ! TODO: ptr to val compare
        ! this wont work:
        ! call assert_equals(f_arr, "howdy")
        !
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_as_char8_str

    
    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_char8_str
        type(C_PTR) cnode
        character, pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_char8_str")
        !----------------------------------------------------------------------

        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        cnode = conduit_node_create()
        ! n = "howdy");
        call conduit_node_set_path_char8_str(cnode,"my_sub","howdy")
        ! n.print();
        call conduit_node_print(cnode)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! char * res = n.as_char8_str();
        call conduit_node_fetch_path_as_char8_str(cnode,"my_sub",f_arr)
        print *, f_arr
        !
        ! TODO: ptr to val compare
        ! this wont work:
        ! call assert_equals(f_arr, "howdy")
        !
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_and_fetch_path_char8_str

!------------------------------------------------------------------------------
end module f_conduit_node_char8_str
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program fortran_test
!------------------------------------------------------------------------------
  use fruit
  use f_conduit_node_char8_str
  implicit none
  logical ok
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_node_set_char8_str
  call t_node_as_char8_str
  call t_node_set_and_fetch_path_char8_str

  
  call fruit_summary
  call fruit_finalize
  call is_all_successful(ok)

  if (.not. ok) then
     call exit(1)
  endif

!------------------------------------------------------------------------------
end program fortran_test
!------------------------------------------------------------------------------

