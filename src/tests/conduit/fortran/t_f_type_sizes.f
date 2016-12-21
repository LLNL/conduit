!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!
!* Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
!* 
!* Produced at the Lawrence Livermore National Laboratory
!* 
!* LLNL-CODE-666778
!* 
!* All rights reserved.
!* 
!* This file is part of Conduit. 
!* 
!* For details, see: http://software.llnl.gov/conduit/.
!* 
!* Please also read conduit/LICENSE
!* 
!* Redistribution and use in source and binary forms, with or without 
!* modification, are permitted provided that the following conditions are met:
!* 
!* * Redistributions of source code must retain the above copyright notice, 
!*   this list of conditions and the disclaimer below.
!* 
!* * Redistributions in binary form must reproduce the above copyright notice,
!*   this list of conditions and the disclaimer (as noted below) in the
!*   documentation and/or other materials provided with the distribution.
!* 
!* * Neither the name of the LLNS/LLNL nor the names of its contributors may
!*   be used to endorse or promote products derived from this software without
!*   specific prior written permission.
!* 
!* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
!* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
!* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
!* ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
!* LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
!* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
!* DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
!* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
!* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
!* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
!* IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
!* POSSIBILITY OF SUCH DAMAGE.
!* 
!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!

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
        print *,""
   
        check_kind = SIZEOF(test_short)
        check_c    = SIZEOF(test_int_2)
        print *,"[SIZEOF] C_SHORT         = ", check_kind
        print *,"[SIZEOF] integer(kind=2) = ",check_c
        print *,""

        check_kind = SIZEOF(test_int)
        check_c    = SIZEOF(test_int_4)
        print *,"[SIZEOF] C_INT           = ", check_kind
        print *,"[SIZEOF] integer(kind=4) = ",check_c
        print *,""

        check_kind = SIZEOF(test_long)
        check_c    = SIZEOF(test_int_8)
        print *,"[SIZEOF] C_LONG          = ", check_kind
        print *,"[SIZEOF] integer(kind=8) = ",check_c
        print *,""

        check_kind = SIZEOF(test_long_long)
        check_c    = SIZEOF(test_int_8)
        print *,"[SIZEOF] C_LONG_LONG     = ", check_kind
        print *,"[SIZEOF] integer(kind=8) = ",check_c
        print *,""

        check_kind = SIZEOF(test_float)
        check_c    = SIZEOF(test_re_4)
        print *,"[SIZEOF] C_FLOAT         = ", check_kind
        print *,"[SIZEOF] real(kind=4)    = ",check_c
        print *,""

        check_kind = SIZEOF(test_double)
        check_c    = SIZEOF(test_re_8)
        print *,"[SIZEOF] C_DOUBLE        = ", check_kind
        print *,"[SIZEOF] real(kind=8)    = ",check_c
        print *,""

        
    end subroutine t_type_sizes


!------------------------------------------------------------------------------
end module f_type_sizes
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
integer(C_INT) function fortran_test() bind(C,name="fortran_test")
!------------------------------------------------------------------------------
  use fruit
  use f_type_sizes
  implicit none
  logical res
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_type_sizes
  
  call fruit_summary
  call fruit_finalize
  call is_all_successful(res)
  if (res) then
     fortran_test = 0
  else
     fortran_test = 1
  endif

!------------------------------------------------------------------------------
end function fortran_test
!------------------------------------------------------------------------------


