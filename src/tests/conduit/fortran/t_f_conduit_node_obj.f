!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!
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
!* For details, see: http://llnl.github.io/conduit/.
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
  use conduit_obj
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!  Obj Style Tests
!------------------------------------------------------------------------------
    !--------------------------------------------------------------------------
    subroutine t_node_obj_create
        type(node) obj
        integer res
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        ! n.print_detailed();
        obj = conduit_node_obj_create()
        call obj%print_detailed()
        call conduit_node_obj_destroy(obj)
    
    end subroutine t_node_obj_create

    !--------------------------------------------------------------------------
    subroutine t_node_obj_set_int
        type(node) obj
        integer res
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;    
        obj = conduit_node_obj_create()
        ! n.set(42);
        call obj%set_int(42)
        ! n.print_detailed();
        call obj%print_detailed()
        ! int res = n.as_int();
        res = obj%as_int()
        call assert_equals (42, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_set_int


    !--------------------------------------------------------------------------
    subroutine t_node_obj_set_int32
        type(node) obj
        integer(4) res
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        obj = conduit_node_obj_create()
        ! n.set_int32(42);
        call obj%set_int32(42)
        ! n.print_detailed();
        call obj%print_detailed()
        ! int32 res = n.as_int32();
        res = obj%as_int32()
        call assert_equals (42, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_set_int32

    !--------------------------------------------------------------------------
    subroutine t_node_obj_set_double
        type(node) obj
        real(kind=8) res
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        obj = conduit_node_obj_create()
        ! n.set(3.1415);
        call obj%set_double(3.1415d+0)
        ! n.print_detailed();
        call obj%print_detailed()
        ! double res = n.as_double();
        res = obj%as_double()
        call assert_equals(3.1415d+0, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_set_double

    !--------------------------------------------------------------------------
    subroutine t_node_obj_set_float64
        type(node) obj
        real(kind=8) res
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        obj = conduit_node_obj_create()
        ! n.set_float64(3.1415);
        call obj%set_float64(3.1415d+0)
        ! n.print_detailed();
        call obj%print_detailed()
        ! float64 res = n.as_float64();
        res = obj%as_float64()
        call assert_equals(3.1415d+0, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_set_float64

    !--------------------------------------------------------------------------
    subroutine t_node_obj_set_int32_ptr
        type(node) obj
        integer(4), dimension(5) :: data
        integer nele
        integer i
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        obj = conduit_node_obj_create()
        call obj%print_detailed()
        ! Node n.set_int32_ptr(data,5);
        call obj%set_int32_ptr(data,5_8)
        ! n.print_detailed();
        call obj%print_detailed()
        ! index_t nele = n.dtype().number_of_elements();
        nele = obj%number_of_elements()
        call assert_equals(nele,5)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_set_int32_ptr
    
    !--------------------------------------------------------------------------
    subroutine t_node_obj_set_external_int32_ptr
        type(node) obj
        integer(4), dimension(5) :: data
        integer res
        integer i
        integer(4), pointer :: f_arr(:)

        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;          
        obj = conduit_node_obj_create()
        ! n.set_external_int32_ptr(data,5);
        call obj%set_external_int32_ptr(data,5_8)
        ! change the first element in the array
        ! so we can check the external semantics
        data(1) = 42
        ! n.print_detailed();
        call obj%print_detailed()
        ! int32 res = n.as_int32()
        res = obj%as_int32()
        call assert_equals(res,42)
        ! int32 *res_ptr = n.as_int32_ptr()
        call obj%as_int32_ptr(f_arr)
        
        call assert_equals(f_arr(1),42)
        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo

        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_set_external_int32_ptr
    
    !--------------------------------------------------------------------------
    subroutine t_node_obj_as_int32_ptr
        type(node) obj
        integer(4), dimension(5) :: data
        integer nele
        integer i
        integer(4), pointer :: f_arr(:)
        
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        obj = conduit_node_obj_create()
        ! n.set_external_int32_ptr(data,5);
        call obj%set_int32_ptr(data,5_8)
        ! n.print_detailed();
        call obj%print_detailed()
        ! index_t nele = n.dtype().number_of_elements();
        nele = obj%number_of_elements()
        call assert_equals(nele,5)
        ! int32 *res_ptr = n.as_int32_ptr()
        call obj%as_int32_ptr(f_arr)
        
        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo
        
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_as_int32_ptr
    
    !--------------------------------------------------------------------------
    subroutine t_node_obj_as_int32_ptr_read_scalar
        type(node) obj
        integer nele
        integer i
        integer(4), pointer :: f_arr(:)

        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        obj = conduit_node_obj_create()
        ! n.set_int32(42);
        call obj%set_int32(42)
        ! n.print_detailed();
        call obj%print_detailed()
        
        ! index_t nele = n.dtype().number_of_elements();
        nele = obj%number_of_elements()
        call assert_equals(nele,1)
        ! int32 *ptr = n.as_int32_ptr();
        call obj%as_int32_ptr(f_arr)
        ! check if ptr[0] == 42
        call assert_equals(f_arr(1),42)

        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_as_int32_ptr_read_scalar
    
    !--------------------------------------------------------------------------
    subroutine t_node_obj_fetch_int32
        type(node) obj
        type(node) n1
        integer res
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        obj = conduit_node_obj_create()
        ! Node &my_sub = n.fetch("my_sub");
        ! //or
        ! Node &my_sub = n["my_sub"];
        n1 = obj%fetch("my_sub")
        ! my_sub.set_int32(42)
        call n1%set_int32(42)
        ! n.print_detailed();
        call obj%print_detailed()
        ! int32 res = n.as_int32();
        res = n1%as_int32()
        call assert_equals (42, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_fetch_int32

    !--------------------------------------------------------------------------
    subroutine t_node_obj_set_fetch_path_int32
        type(node) obj
        type(node) n1
        integer res
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        obj = conduit_node_obj_create()
        ! n["my_sub"].set_int32(42)
        ! // or
        ! n.set_path_int32("my_sub",42)
        call obj%set_path_int32("my_sub",42)
        ! n.print_detailed()
        call obj%print_detailed()

        ! int32 res = n["my_sub"].as_int32();
        res = obj%fetch_path_as_int32("my_sub")
        call assert_equals(42, res)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_set_fetch_path_int32
    
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

        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        obj = conduit_node_obj_create()
        ! Node &n1 = n.append();
        n1 = obj%append()
        ! Node &n2 = n.append();
        n2 = obj%append()
        ! index_t nchld = n.number_of_children();
        nchld = obj%number_of_children()
        
        call assert_equals(nchld, 2)
        
        ! n1.set_int32(42);
        call n1%set_int32(42)
        ! n1.set_float64(3.1415);
        call n2%set_float64(3.1415d+0)
        
        ! n.print_detailed();
        call obj%print_detailed()
        
        ! Node &na = n[0];
        ! // or
        ! Node &na = n.child(0);
        na  = obj%child(0_8)
        ! Node &nb = n[1];
        ! // or
        ! Node &nb = n.child(1);
        nb  = obj%child(1_8)
        
        call obj%print_detailed()

        !int32 res_1 = n.as_int32();
        res_1 = n1%as_int32()
        !int32 res_2 = n.as_float64();
        res_2 = n2%as_float64()
        
        call assert_equals (42, res_1)
        call assert_equals (3.1415d+0, res_2)
        call conduit_node_obj_destroy(obj)
        
    end subroutine t_node_obj_append


!------------------------------------------------------------------------------
end module f_conduit_node
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
integer(C_INT) function fortran_test() bind(C,name="fortran_test")
!------------------------------------------------------------------------------
  use fruit
  use f_conduit_node
  implicit none
  logical res
  call init_fruit
  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_node_obj_create
  call t_node_obj_set_int
  call t_node_obj_set_int32
  call t_node_obj_set_double
  call t_node_obj_set_float64
  call t_node_obj_fetch_int32
  call t_node_obj_set_int32_ptr
  call t_node_obj_set_external_int32_ptr
  call t_node_obj_append
  call t_node_obj_set_fetch_path_int32

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
