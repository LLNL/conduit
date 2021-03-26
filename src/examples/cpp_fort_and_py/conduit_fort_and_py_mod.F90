! Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
! Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
! other details. No copyright assignment is required to contribute to Conduit.

!-----------------------------------------------------------------------------
!
! file: conduit_fort_and_py_mod.F90
!
!/-----------------------------------------------------------------------------


!-----------------------------------------------------------------------------
!
! fortran binding interface for conduit_fort_and_py_mod module
! 
!-----------------------------------------------------------------------------

module conduit_fort_and_py_mod
!------------------------------------------------------------------------------
    use, intrinsic :: iso_c_binding, only : C_PTR
    implicit none

    interface
!--------------------------------------------------------------------------
    subroutine conduit_fort_to_py(cnode) &
        bind(C, name="conduit_fort_to_py")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
    end subroutine conduit_fort_to_py

    !--------------------------------------------------------------------------
    function c_conduit_fort_from_py(name) result(res) &
             bind(C, name="conduit_fort_from_py")
         use iso_c_binding
         implicit none
         character(kind=C_CHAR), intent(IN) :: name(*)
         type(C_PTR) :: res
     end function c_conduit_fort_from_py


    end interface

contains

    !--------------------------------------------------------------------------
    ! Note this method exists to apply fortran trim to passed string
    ! before passing to the c api
    !--------------------------------------------------------------------------
    function conduit_fort_from_py(name) result(res)
        use iso_c_binding
        implicit none
        character(*), intent(IN) :: name
        type(C_PTR) :: res
        !---
        res = c_conduit_fort_from_py(trim(name) // C_NULL_CHAR)
    end function conduit_fort_from_py

!------------------------------------------------------------------------------
end module conduit_fort_and_py_mod
!------------------------------------------------------------------------------
