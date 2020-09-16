! Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
! Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
! other details. No copyright assignment is required to contribute to Conduit.


!------------------------------------------------------------------------------
! fortran_test_obj_support.f
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
! Basic check for fortran compiler support of the "class" specifier
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module f_obj_test
    !--------------------------------------------------------------------------
    type obj
    contains

        !----------------------------------------------------------------------
        procedure :: go  => obj_go

    end type obj
    !--------------------------------------------------------------------------

!------------------------------------------------------------------------------
contains

    !--------------------------------------------------------------------------
    function obj_go(o) result(v)
        implicit none
        class(obj) :: o
        integer(4) :: v
        v = 42
    end function obj_go

end module
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program test
    use f_obj_test
    type(obj):: t
    integer(4)::v 
    v = t%go()
    print *, "Test:", v
end program test


