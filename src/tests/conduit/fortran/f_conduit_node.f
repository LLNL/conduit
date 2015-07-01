!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!*
!* Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
!* 
!* Produced at the Lawrence Livermore National Laboratory
!* 
!* LLNL-CODE-666778
!* 
!* All rights reserved.
!* 
!* This file is part of Conduit. 
!* 
!* For details, see https://lc.llnl.gov/conduit/.
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
!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!*

!------------------------------------------------------------------------------
!
! f_conduit_node.f
!
!------------------------------------------------------------------------------
module f_conduit_node
  use iso_c_binding
  use fruit
  use conduit
  implicit none

contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------    
!  Opaque Pointer Function Style test
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_node_create
        type(C_PTR) cnode
        integer res

        cnode = conduit_node_create()
        call conduit_node_print_detailed(cnode)
        call conduit_node_destroy(cnode)
    
    end subroutine t_node_create

    !--------------------------------------------------------------------------
    subroutine t_node_set_int
        type(C_PTR) cnode
        integer res
    
        cnode = conduit_node_create()
        call conduit_node_set_int(cnode,42)
        call conduit_node_print_detailed(cnode)
        res = conduit_node_as_int(cnode)
        call assert_equals (42, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_int

    !--------------------------------------------------------------------------
    subroutine t_node_set_int32
        type(C_PTR) cnode
        integer(4) res
        
        cnode = conduit_node_create()
        call conduit_node_set_int32(cnode,42)
        call conduit_node_print_detailed(cnode)
        res = conduit_node_as_int32(cnode)
        call assert_equals (42, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_int32

    !--------------------------------------------------------------------------
    subroutine t_node_set_double
        type(C_PTR) cnode
        real(kind=8) res
        
        cnode = conduit_node_create()
        call conduit_node_set_double(cnode,3.1415d+0)
        call conduit_node_print_detailed(cnode)
        res = conduit_node_as_double(cnode)
        call assert_equals(3.1415d+0, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_double

    !--------------------------------------------------------------------------
    subroutine t_node_set_float64
        type(C_PTR) cnode
        real(kind=8) res
        
        cnode = conduit_node_create()
        call conduit_node_set_float64(cnode,3.1415d+0)
        call conduit_node_print_detailed(cnode)
        res = conduit_node_as_float64(cnode)
        call assert_equals(3.1415d+0, res)
        call conduit_node_destroy(cnode)
                
    end subroutine t_node_set_float64

    !--------------------------------------------------------------------------
    subroutine t_node_set_int32_ptr
        type(C_PTR) cnode
        integer(4), dimension(5) :: data
        integer i
        
        do i = 1,5
            data(i) = i
        enddo
         
        cnode = conduit_node_create()
        call conduit_node_set_int32_ptr(cnode,data,5_8)
        call conduit_node_print_detailed(cnode)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_int32_ptr

    !--------------------------------------------------------------------------
    subroutine t_node_fetch_int32
        type(C_PTR) cnode
        type(C_PTR) n1
        integer res
        
        cnode = conduit_node_create()
        
        n1 = conduit_node_fetch(cnode,"my_sub")
        call conduit_node_set_int32(n1,42)
        
        call conduit_node_print_detailed(cnode)
        
        res = conduit_node_as_int32(n1)
        call assert_equals (42, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_fetch_int32

    !--------------------------------------------------------------------------
    subroutine t_node_set_fetch_path_int32
        type(C_PTR) cnode
        type(C_PTR) n1
        integer res
        
        cnode = conduit_node_create()

        call conduit_node_set_path_int32(cnode,"my_sub",42)
        call conduit_node_print_detailed(cnode)

        res = conduit_node_fetch_path_as_int32(cnode,"my_sub")
        call assert_equals (42, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_fetch_path_int32

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
        
        cnode = conduit_node_create()
        
        n1 = conduit_node_append(cnode)
        n2 = conduit_node_append(cnode)
        
        nchld = conduit_node_number_of_children(cnode)
        
        call assert_equals(nchld, 2)
        
        call conduit_node_set_int32(n1,42)
        call conduit_node_set_float64(n2,3.1415d+0)
        
        call conduit_node_print_detailed(cnode)
        
        na  = conduit_node_child(cnode,0_8)
        nb  = conduit_node_child(cnode,1_8)
                        
        res_1 = conduit_node_as_int32(n1)
        res_2 = conduit_node_as_float64(n2)
        
        call assert_equals (42, res_1)
        call assert_equals (3.1415d+0, res_2)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_append


!----------------------------------------------------------------------
end module f_conduit_node
!----------------------------------------------------------------------

function fortran_test() bind(C,name="fortran_test")
  use fruit
  use f_conduit_node
  implicit none
  integer(C_INT) fortran_test

  call init_fruit

  call t_node_create
  call t_node_set_int
  call t_node_set_int32
  call t_node_set_double
  call t_node_set_float64
  call t_node_fetch_int32
  call t_node_set_int32_ptr
  call t_node_set_fetch_path_int32
  call t_node_append
  
  call fruit_summary
  call fruit_finalize

  fortran_test = 0
end function fortran_test

