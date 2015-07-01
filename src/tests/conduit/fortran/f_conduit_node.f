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
!* For details, see https:!*lc.llnl.gov/conduit/.
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
    subroutine t_node_fetch_int32
        type(node) obj
        type(node) n1
        integer res
        
        obj = conduit_node_obj_create()
        
        n1 = obj%fetch("my_sub")
        call n1%set_int32(42)
        
        call obj%print_detailed()
        
        res = n1%as_int32()
        call assert_equals (42, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_fetch_int32

    !--------------------------------------------------------------------------
    subroutine t_node_append
        type(node) obj
        type(node) n1
        type(node) n2
        type(node) na
        type(node) nb
        integer(4) res_1
        real(8)    res_2
        integer    nchld
        
        obj = conduit_node_obj_create()
        
        n1 = obj%append()
        n2 = obj%append()
        
        nchld = obj%number_of_children()
        
        call assert_equals(nchld, 2)
        
        call n1%set_int32(42)
        call n2%set_float64(3.1415d+0)
        
        call obj%print_detailed()
        
        ! TODO: these crash?
        na  = obj%child(0_8)
        nb  = obj%child(1_8)
        
        call obj%print_detailed()
                
        res_1 = n1%as_int32()
        res_2 = n2%as_float64()
        
        call assert_equals (42, res_1)
        call assert_equals (3.1415d+0, res_2)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_append


    !--------------------------------------------------------------------------
    subroutine t_node_set_int32
        type(node) obj
        integer(4) res
        
        obj = conduit_node_obj_create()
        call obj%set_int32(42)
        call obj%print_detailed()
        res = obj%as_int32()
        call assert_equals (42, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_set_int32

    !--------------------------------------------------------------------------
    subroutine t_node_set_double
        type(node) obj
        real(kind=8) res
        
        obj = conduit_node_obj_create()
        call obj%set_double(3.1415d+0)
        call obj%print_detailed()
        res = obj%as_double()
        call assert_equals(3.1415d+0, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_set_double

    !--------------------------------------------------------------------------
    subroutine t_node_set_float64
        type(node) obj
        real(kind=8) res
        
        obj = conduit_node_obj_create()
        call obj%set_float64(3.1415d+0)
        call obj%print_detailed()
        res = obj%as_float64()
        call assert_equals(3.1415d+0, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_set_float64

    !--------------------------------------------------------------------------
    subroutine t_node_set_int32_ptr
        type(node) obj
        integer(4), dimension(5) :: data
        integer i
        
        do i = 1,5
            data(i) = i
        enddo
         
        obj = conduit_node_obj_create()
        call obj%print_detailed()
        call obj%set_int32_ptr(data,5_8)
        call obj%print_detailed()
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_set_int32_ptr


!------------------------------------------------------------------------------
!  Obj Style Tests
!------------------------------------------------------------------------------
    !--------------------------------------------------------------------------
    subroutine t_node_obj_create
        type(node) obj
        integer res
    
        obj = conduit_node_obj_create()
        call obj%print_detailed()
        call conduit_node_obj_destroy(obj)
    
    end subroutine t_node_obj_create

    !--------------------------------------------------------------------------
    subroutine t_node_obj_set_int
        type(node) obj
        integer res
        
        obj = conduit_node_obj_create()
        call obj%set_int(42)
        call obj%print_detailed()
        res = obj%as_int()
        call assert_equals (42, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_set_int

    !--------------------------------------------------------------------------
    subroutine t_node_obj_fetch_int32
        type(node) obj
        type(node) n1
        integer res
        
        obj = conduit_node_obj_create()
        
        n1 = obj%fetch("my_sub")
        call n1%set_int32(42)
        
        call obj%print_detailed()
        
        res = n1%as_int32()
        call assert_equals (42, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_fetch_int32

    !--------------------------------------------------------------------------
    subroutine t_node_obj_append
        type(node) obj
        type(node) n1
        type(node) n2
        type(node) na
        type(node) nb
        integer(4) res_1
        real(8)    res_2
        integer    nchld
        
        obj = conduit_node_obj_create()
        
        n1 = obj%append()
        n2 = obj%append()
        
        nchld = obj%number_of_children()
        
        call assert_equals(nchld, 2)
        
        call n1%set_int32(42)
        call n2%set_float64(3.1415d+0)
        
        call obj%print_detailed()
        
        ! TODO: these crash?
        na  = obj%child(0_8)
        nb  = obj%child(1_8)
        
        call obj%print_detailed()
                
        res_1 = n1%as_int32()
        res_2 = n2%as_float64()
        
        call assert_equals (42, res_1)
        call assert_equals (3.1415d+0, res_2)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_append


    !--------------------------------------------------------------------------
    subroutine t_node_obj_set_int32
        type(node) obj
        integer(4) res
        
        obj = conduit_node_obj_create()
        call obj%set_int32(42)
        call obj%print_detailed()
        res = obj%as_int32()
        call assert_equals (42, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_set_int32

    !--------------------------------------------------------------------------
    subroutine t_node_obj_set_double
        type(node) obj
        real(kind=8) res
        
        obj = conduit_node_obj_create()
        call obj%set_double(3.1415d+0)
        call obj%print_detailed()
        res = obj%as_double()
        call assert_equals(3.1415d+0, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_set_double

    !--------------------------------------------------------------------------
    subroutine t_node_obj_set_float64
        type(node) obj
        real(kind=8) res
        
        obj = conduit_node_obj_create()
        call obj%set_float64(3.1415d+0)
        call obj%print_detailed()
        res = obj%as_float64()
        call assert_equals(3.1415d+0, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_set_float64

    !--------------------------------------------------------------------------
    subroutine t_node_obj_set_int32_ptr
        type(node) obj
        integer(4), dimension(5) :: data
        integer i
        
        do i = 1,5
            data(i) = i
        enddo
         
        obj = conduit_node_obj_create()
        call obj%print_detailed()
        call obj%set_int32_ptr(data,5_8)
        call obj%print_detailed()
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_set_int32_ptr


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
  call t_node_append
  call t_node_set_int32_ptr

  call t_node_obj_create
  call t_node_obj_set_int
  call t_node_obj_set_int32
  call t_node_obj_set_double
  call t_node_obj_set_float64
  call t_node_obj_fetch_int32
  call t_node_obj_append
  call t_node_obj_set_int32_ptr

  
  call fruit_summary
  call fruit_finalize

  fortran_test = 0
end function fortran_test

