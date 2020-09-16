!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.

!------------------------------------------------------------------------------
!
! t_f_conduit_smoke.f
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module t_f_conduit_smoke
!------------------------------------------------------------------------------

  use iso_c_binding
  use fruit
  use conduit
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------    
! About test
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_conduit_about
        type(C_PTR) cnode
        integer res
        
        !----------------------------------------------------------------------
        call set_case_name("t_conduit_about")
        !----------------------------------------------------------------------
        
        cnode = conduit_node_create()
        call conduit_about(cnode)
        call conduit_node_print(cnode)
        call conduit_node_destroy(cnode)
    
    end subroutine t_conduit_about

!------------------------------------------------------------------------------
end module t_f_conduit_smoke
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program fortran_test
!------------------------------------------------------------------------------
  use fruit
  use t_f_conduit_smoke
  implicit none
  logical ok
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_conduit_about

  call fruit_summary
  call fruit_finalize
  call is_all_successful(ok)
  
  if (.not. ok) then
     call exit(1)
  endif

!------------------------------------------------------------------------------
end program fortran_test
!------------------------------------------------------------------------------


