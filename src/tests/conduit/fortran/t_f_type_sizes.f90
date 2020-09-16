!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.

!------------------------------------------------------------------------------
!
! f_type_sizes.f
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module f_type_sizes
!------------------------------------------------------------------------------

  use iso_c_binding
  use fruit
  use conduit
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    ! Note: This is more of a diagnostic than a test.
    !--------------------------------------------------------------------------
    ! Even though there is diversity with respect to the use of "kind" among 
    ! fortran compilers, we will use the kind variants as maps to our
    ! bitwidth style types. 
    !--------------------------------------------------------------------------
    subroutine t_type_sizes
        !---- 
        integer(kind=1) test_int_1
        integer(kind=2) test_int_2
        integer(kind=4) test_int_4
        integer(kind=8) test_int_8
        real(kind=4) test_re_4
        real(kind=8) test_re_8
        !---
        integer(kind=C_SIGNED_CHAR) test_char
        integer(kind=C_SHORT)       test_short
        integer(kind=C_INT)         test_int
        integer(kind=C_LONG)        test_long
        integer(kind=C_LONG_LONG)   test_long_long
        real(kind=C_FLOAT)  test_float
        real(kind=C_DOUBLE) test_double
        !---
        integer check_kind, check_c

        !----------------------------------------------------------------------
        call set_case_name("t_type_sizes")
        !----------------------------------------------------------------------


        check_kind = SIZEOF(test_char)
        check_c    = SIZEOF(test_int_1)
        print *,"[SIZEOF] C_SIGNED_CHAR   = ", check_kind
        print *,"[SIZEOF] integer(kind=1) = ",check_c
        print *," "
   
        check_kind = SIZEOF(test_short)
        check_c    = SIZEOF(test_int_2)
        print *,"[SIZEOF] C_SHORT         = ", check_kind
        print *,"[SIZEOF] integer(kind=2) = ",check_c
        print *," "

        check_kind = SIZEOF(test_int)
        check_c    = SIZEOF(test_int_4)
        print *,"[SIZEOF] C_INT           = ", check_kind
        print *,"[SIZEOF] integer(kind=4) = ",check_c
        print *," "

        check_kind = SIZEOF(test_long)
        check_c    = SIZEOF(test_int_8)
        print *,"[SIZEOF] C_LONG          = ", check_kind
        print *,"[SIZEOF] integer(kind=8) = ",check_c
        print *," "

        check_kind = SIZEOF(test_long_long)
        check_c    = SIZEOF(test_int_8)
        print *,"[SIZEOF] C_LONG_LONG     = ", check_kind
        print *,"[SIZEOF] integer(kind=8) = ",check_c
        print *," "

        check_kind = SIZEOF(test_float)
        check_c    = SIZEOF(test_re_4)
        print *,"[SIZEOF] C_FLOAT         = ", check_kind
        print *,"[SIZEOF] real(kind=4)    = ",check_c
        print *," "

        check_kind = SIZEOF(test_double)
        check_c    = SIZEOF(test_re_8)
        print *,"[SIZEOF] C_DOUBLE        = ", check_kind
        print *,"[SIZEOF] real(kind=8)    = ",check_c
        print *," "

        
    end subroutine t_type_sizes


!------------------------------------------------------------------------------
end module f_type_sizes
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program fortran_test
!------------------------------------------------------------------------------
  use fruit
  use f_type_sizes
  implicit none
  logical ok
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_type_sizes
  
  call fruit_summary
  call fruit_finalize
  call is_all_successful(ok)
  
  if (.not. ok) then
     call exit(1)
  endif

!------------------------------------------------------------------------------
end program fortran_test
!------------------------------------------------------------------------------


