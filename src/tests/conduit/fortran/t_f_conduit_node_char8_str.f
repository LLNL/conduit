!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!
!* Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
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

