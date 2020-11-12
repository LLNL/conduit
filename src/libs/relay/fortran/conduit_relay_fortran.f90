!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.

!------------------------------------------------------------------------------
! conduit_relay_fortran.F
!------------------------------------------------------------------------------


!------------------------------------------------------------------------------
module conduit_relay
!------------------------------------------------------------------------------
    use, intrinsic :: iso_c_binding, only : C_PTR
    implicit none

    !--------------------------------------------------------------------------
    interface
    !--------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine conduit_relay_about(cnode) &
             bind(C, name="conduit_relay_about")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: cnode
     end subroutine conduit_relay_about

     !--------------------------------------------------------------------------
     end interface
     !--------------------------------------------------------------------------

!------------------------------------------------------------------------------
end module conduit_relay
!------------------------------------------------------------------------------

