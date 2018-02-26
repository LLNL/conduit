!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!
!* Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
! f_conduit_node.f
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module f_conduit_node
!------------------------------------------------------------------------------

  use iso_c_binding
  use fruit
  use conduit
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------    
!  Opaque Pointer Function Style test
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_node_create
        type(C_PTR) cnode
        integer res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_create")
        !----------------------------------------------------------------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        ! n.print_detailed();
        cnode = conduit_node_create()
        call assert_true(conduit_node_is_root(cnode) .eqv. .true. )
        call conduit_node_print_detailed(cnode)
        call conduit_node_destroy(cnode)
    
    end subroutine t_node_create


    !--------------------------------------------------------------------------
    subroutine t_node_append
        type(C_PTR) cnode
        type(C_PTR) n1
        type(C_PTR) n2
        type(C_PTR) na
        type(C_PTR) nb
        integer(4) res_1
        real(8)    res_2
        integer    nchld
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_append")
        !----------------------------------------------------------------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;  
        cnode = conduit_node_create()
        
        ! Node &n1 = n.append();
        n1 = conduit_node_append(cnode)
        ! Node &n2 = n.append();
        n2 = conduit_node_append(cnode)

        call assert_true( conduit_node_is_root(n2) .eqv. .false.)
        
        ! index_t nchld = n.number_of_children();
        nchld = conduit_node_number_of_children(cnode)
        
        call assert_equals(nchld, 2)
        ! n1.set_int32(42);
        call conduit_node_set_int32(n1,42)
        ! n1.set_float64(3.1415);
        call conduit_node_set_float64(n2,3.1415d+0)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        
        ! Node &na = n[0];
        ! // or
        ! Node &na = n.child(0);
        na  = conduit_node_child(cnode,0_8)
        ! Node &nb = n[1];
        ! // or
        ! Node &nb = n.child(1);
        nb  = conduit_node_child(cnode,1_8)

        !int32 res_1 = n.as_int32();
        res_1 = conduit_node_as_int32(n1)
        !int32 res_2 = n.as_float64();
        res_2 = conduit_node_as_float64(n2)
        
        call assert_equals (42, res_1)
        call assert_equals (3.1415d+0, res_2)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_append


    !--------------------------------------------------------------------------
    subroutine t_node_set_int
        type(C_PTR) cnode
        integer res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_int")
        !----------------------------------------------------------------------

        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;    
        cnode = conduit_node_create()
        ! n.set(42);
        call conduit_node_set_int(cnode,42)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! int res = n.as_int();
        res = conduit_node_as_int(cnode)
        call assert_equals (42, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_int

    !--------------------------------------------------------------------------
    subroutine t_node_set_double
        type(C_PTR) cnode
        real(kind=8) res
        
        !--------------
        call set_case_name("t_node_set_double")
        !--------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        cnode = conduit_node_create()
        ! n.set(3.1415);
        call conduit_node_set_double(cnode,3.1415d+0)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! double res = n.as_double();
        res = conduit_node_as_double(cnode)
        call assert_equals(3.1415d+0, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_double

    !--------------------------------------------------------------------------
    subroutine t_node_set_float64
        type(C_PTR) cnode
        real(kind=8) res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_float64")
        !----------------------------------------------------------------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        cnode = conduit_node_create()
        ! n.set_float64(3.1415);
        call conduit_node_set_float64(cnode,3.1415d+0)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! float64 res = n.as_float64();
        res = conduit_node_as_float64(cnode)
        call assert_equals(3.1415d+0, res)
        call conduit_node_destroy(cnode)

    end subroutine t_node_set_float64

    !--------------------------------------------------------------------------
    subroutine t_node_diff
        type(C_PTR) cnode1
        type(C_PTR) cnode2
        type(C_PTR) cinfo
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_diff")
        !----------------------------------------------------------------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n1;
        ! Node n2;
        ! Node info;
        cnode1 = conduit_node_create()
        cnode2 = conduit_node_create()
        cinfo  = conduit_node_create()

        ! n1["a"].set_float64(3.1415);
        call conduit_node_set_path_float64(cnode1,"a",3.1415d+0)
        ! n1.diff(n2,info,1e-12)
        !! there is a diff
        call assert_true( conduit_node_diff(cnode1,cnode2,cinfo,1d-12) .eqv. .true.)
        ! n2["a"].set_float64(3.1415);
        call conduit_node_set_path_float64(cnode2,"a",3.1415d+0)
        ! n1.diff(n2,info,1e-12)
        !! no diff
        call assert_true( conduit_node_diff(cnode1,cnode2,cinfo,1d-12) .eqv. .false.)
        ! n2["b"].set_float64(3.1415);
        call conduit_node_set_path_float64(cnode2,"b",3.1415d+0)
        ! n1.diff(n2,info,1e-12)
        !! there is a diff
        call assert_true( conduit_node_diff(cnode1,cnode2,cinfo,1d-12) .eqv. .true.)
        
        ! n1.diff(n2,info,1e-12)
        !! but no diff compat
        call assert_true( conduit_node_diff_compatible(cnode1,cnode2,cinfo,1d-12) .eqv. .false.)
        
        call conduit_node_destroy(cnode1)
        call conduit_node_destroy(cnode2)
        call conduit_node_destroy(cinfo)

    end subroutine t_node_diff


!------------------------------------------------------------------------------
end module f_conduit_node
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program fortran_test
!------------------------------------------------------------------------------
  use fruit
  use f_conduit_node
  implicit none
  logical ok
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_node_create
  call t_node_append
  call t_node_set_int
  call t_node_set_double
  call t_node_set_float64
  call t_node_diff

  
  call fruit_summary
  call fruit_finalize
  call is_all_successful(ok)
  
  if (.not. ok) then
     call exit(1)
  endif

!------------------------------------------------------------------------------
end program fortran_test
!------------------------------------------------------------------------------


