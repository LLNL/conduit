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
module f_conduit_node_float64
!------------------------------------------------------------------------------

  use iso_c_binding
  use fruit
  use conduit
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------    
! float32 tests
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_node_set_float32
        type(C_PTR) cnode
        real(4) res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_float32")
        !----------------------------------------------------------------------

        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        cnode = conduit_node_create()
        ! n.set_float32(3.1415);
        call conduit_node_set_float32(cnode,3.1415)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! float32 res = n.as_float32();
        res = conduit_node_as_float32(cnode)
        call assert_equals (3.1415, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_float32

    !--------------------------------------------------------------------------
    subroutine t_node_set_float32_ptr
        type(C_PTR) cnode
        real(4), dimension(5) :: data
        integer nele
        integer i
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_float32_ptr")
        !----------------------------------------------------------------------
        
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! Node n.set_float32_ptr(data,5);
        call conduit_node_set_float32_ptr(cnode,data,5_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! index_t nele = n.dtype().number_of_elements();
        nele = conduit_node_number_of_elements(cnode)
        call assert_equals(nele,5)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_float32_ptr

    !--------------------------------------------------------------------------
    subroutine t_node_set_external_float32_ptr
        type(C_PTR) cnode
        real(4), dimension(5) :: data
        real(4) res
        integer i
        real(4), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_external_float32_ptr")
        !----------------------------------------------------------------------
        
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_float32_ptr(data,5);
        call conduit_node_set_external_float32_ptr(cnode,data,5_8)
        ! change the first element in the array
        ! so we can check the external semantics
        data(1) = 3.1415
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! float32 res = n.as_float32()
        res = conduit_node_as_float32(cnode)
        call assert_equals(res,3.1415)
        ! float32 *res_ptr = n.as_float32_ptr()
        call conduit_node_as_float32_ptr(cnode,f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));
        
        call assert_equals(f_arr(1),3.1415)
        
        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_external_float32_ptr

    !--------------------------------------------------------------------------
    subroutine t_node_as_float32
        type(C_PTR) cnode
        type(C_PTR) n1
        real(4) res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_float32")
        !----------------------------------------------------------------------
                
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_float32(3.1415)
        call conduit_node_set_float32(cnode,3.1415)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! float32 res = n.as_float32();
        res = conduit_node_as_float32(cnode)
        call assert_equals (3.1415, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_as_float32

    !--------------------------------------------------------------------------
    subroutine t_node_as_float32_ptr
        type(C_PTR) cnode
        real(4), dimension(5) :: data
        integer nele
        integer i
        real(4), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_float32_ptr")
        !----------------------------------------------------------------------
        
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_float32_ptr(data,5);
        call conduit_node_set_float32_ptr(cnode,data,5_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! index_t nele = n.dtype().number_of_elements();
        nele = conduit_node_number_of_elements(cnode)
        call assert_equals(nele,5)
        ! float32 *res_ptr = n.as_float32_ptr()
        call conduit_node_as_float32_ptr(cnode,f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));
        
        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_as_float32_ptr


    !--------------------------------------------------------------------------
    subroutine t_node_as_float32_ptr_read_scalar
        type(C_PTR) cnode
        integer nele
        integer i
        real(4), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_float32_ptr_read_scalar")
        !----------------------------------------------------------------------
                
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_float32(3.1415);
        call conduit_node_set_float32(cnode,3.1415)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! index_t nele = n.dtype().number_of_elements();
        nele = conduit_node_number_of_elements(cnode)
        call assert_equals(nele,1)
        ! float32 *ptr = n.as_float32_ptr();
        call conduit_node_as_float32_ptr(cnode,f_arr)
        ! check if ptr[0] == 3.1415
        call assert_equals(f_arr(1),3.1415)

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_as_float32_ptr_read_scalar


    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_float32
        type(C_PTR) cnode
        type(C_PTR) n1
        real(4) res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_float32")
        !----------------------------------------------------------------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;     
        cnode = conduit_node_create()
        ! n["my_sub"].set_float32(3.1415)
        ! // or
        ! n.set_path_float32("my_sub",3.1415)
        call conduit_node_set_path_float32(cnode,"my_sub",3.1415)
        ! n.print_detailed()
        call conduit_node_print_detailed(cnode)
        ! float32 res = n["my_sub"].as_float32();
        res = conduit_node_fetch_path_as_float32(cnode,"my_sub")
        call assert_equals (3.1415, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_and_fetch_path_float32


    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_float32_ptr
        type(C_PTR) cnode
        real(4), dimension(5) :: data
        real(4) res
        integer i
        real(4), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_float32_ptr")
        !----------------------------------------------------------------------
        
        ! fill our array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_float32_ptr(data,5);
        call conduit_node_set_path_float32_ptr(cnode,"my_sub",data,5_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! float32 *res_ptr = n.as_float32_ptr()
        call conduit_node_fetch_path_as_float32_ptr(cnode,"my_sub",f_arr)
        
        call assert_equals(size(data),size(f_arr));

        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_and_fetch_path_float32_ptr


    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_external_float32_ptr
        type(C_PTR) cnode
        real(4), dimension(5) :: data
        real(4) res
        integer i
        real(4), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_external_float32_ptr")
        !----------------------------------------------------------------------
        
        ! fill our array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_float32_ptr(data,5);
        call conduit_node_set_path_external_float32_ptr(cnode,"my_sub",data,5_8)
        ! change the first element in the array
        ! so we can check the external semantics
        data(1) = 3.1415
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! float32 res = n.as_float32()
        res = conduit_node_fetch_path_as_float32(cnode,"my_sub")
        call assert_equals(res,3.1415)
        ! float32 *res_ptr = n.as_float32_ptr()
        call conduit_node_fetch_path_as_float32_ptr(cnode,"my_sub",f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));
        
        call assert_equals(f_arr(1),3.1415)
        
        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_and_fetch_path_external_float32_ptr
    
!------------------------------------------------------------------------------    
! float64 tests
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_node_set_float64
        type(C_PTR) cnode
        real(8) res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_float64")
        !----------------------------------------------------------------------

        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        cnode = conduit_node_create()
        ! n.set_float64(3.1415d+0);
        call conduit_node_set_float64(cnode,3.1415d+0)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! float64 res = n.as_float64();
        res = conduit_node_as_float64(cnode)
        call assert_equals (3.1415d+0, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_float64

    !--------------------------------------------------------------------------
    subroutine t_node_set_float64_ptr
        type(C_PTR) cnode
        real(8), dimension(5) :: data
        integer nele
        integer i
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_float64_ptr")
        !----------------------------------------------------------------------
        
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! Node n.set_float64_ptr(data,5);
        call conduit_node_set_float64_ptr(cnode,data,5_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! index_t nele = n.dtype().number_of_elements();
        nele = conduit_node_number_of_elements(cnode)
        call assert_equals(nele,5)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_float64_ptr

    !--------------------------------------------------------------------------
    subroutine t_node_set_external_float64_ptr
        type(C_PTR) cnode
        real(8), dimension(5) :: data
        real(8) res
        integer i
        real(8), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_external_float64_ptr")
        !----------------------------------------------------------------------
        
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_float64_ptr(data,5);
        call conduit_node_set_external_float64_ptr(cnode,data,5_8)
        ! change the first element in the array
        ! so we can check the external semantics
        data(1) = 3.1415d+0
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! float64 res = n.as_float64()
        res = conduit_node_as_float64(cnode)
        call assert_equals(res,3.1415d+0)
        ! float64 *res_ptr = n.as_float64_ptr()
        call conduit_node_as_float64_ptr(cnode,f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));
        
        call assert_equals(f_arr(1),3.1415d+0)
        
        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_external_float64_ptr

    !--------------------------------------------------------------------------
    subroutine t_node_as_float64
        type(C_PTR) cnode
        type(C_PTR) n1
        real(8) res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_float64")
        !----------------------------------------------------------------------
                
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_float64(3.1415d+0)
        call conduit_node_set_float64(cnode,3.1415d+0)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! float64 res = n.as_float64();
        res = conduit_node_as_float64(cnode)
        call assert_equals (3.1415d+0, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_as_float64

    !--------------------------------------------------------------------------
    subroutine t_node_as_float64_ptr
        type(C_PTR) cnode
        real(8), dimension(5) :: data
        integer nele
        integer i
        real(8), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_float64_ptr")
        !----------------------------------------------------------------------
        
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_float64_ptr(data,5);
        call conduit_node_set_float64_ptr(cnode,data,5_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! index_t nele = n.dtype().number_of_elements();
        nele = conduit_node_number_of_elements(cnode)
        call assert_equals(nele,5)
        ! float64 *res_ptr = n.as_float64_ptr()
        call conduit_node_as_float64_ptr(cnode,f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));
        
        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_as_float64_ptr


    !--------------------------------------------------------------------------
    subroutine t_node_as_float64_ptr_read_scalar
        type(C_PTR) cnode
        integer nele
        integer i
        real(8), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_float64_ptr_read_scalar")
        !----------------------------------------------------------------------
                
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_float64(3.1415d+0);
        call conduit_node_set_float64(cnode,3.1415d+0)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! index_t nele = n.dtype().number_of_elements();
        nele = conduit_node_number_of_elements(cnode)
        call assert_equals(nele,1)
        ! float64 *ptr = n.as_float64_ptr();
        call conduit_node_as_float64_ptr(cnode,f_arr)
        ! check if ptr[0] == 3.1415d+0
        call assert_equals(f_arr(1),3.1415d+0)

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_as_float64_ptr_read_scalar


    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_float64
        type(C_PTR) cnode
        type(C_PTR) n1
        real(8) res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_float64")
        !----------------------------------------------------------------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;     
        cnode = conduit_node_create()
        ! n["my_sub"].set_float64(3.1415d+0)
        ! // or
        ! n.set_path_float64("my_sub",3.1415d+0)
        call conduit_node_set_path_float64(cnode,"my_sub",3.1415d+0)
        ! n.print_detailed()
        call conduit_node_print_detailed(cnode)
        ! float64 res = n["my_sub"].as_float64();
        res = conduit_node_fetch_path_as_float64(cnode,"my_sub")
        call assert_equals (3.1415d+0, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_and_fetch_path_float64


    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_float64_ptr
        type(C_PTR) cnode
        real(8), dimension(5) :: data
        real(8) res
        integer i
        real(8), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_float64_ptr")
        !----------------------------------------------------------------------
        
        ! fill our array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_float64_ptr(data,5);
        call conduit_node_set_path_float64_ptr(cnode,"my_sub",data,5_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! float64 *res_ptr = n.as_float64_ptr()
        call conduit_node_fetch_path_as_float64_ptr(cnode,"my_sub",f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));

        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo


        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_and_fetch_path_float64_ptr


    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_external_float64_ptr
        type(C_PTR) cnode
        real(8), dimension(5) :: data
        real(8) res
        integer i
        real(8), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_external_float64_ptr")
        !----------------------------------------------------------------------
        
        ! fill our array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_float64_ptr(data,5);
        call conduit_node_set_path_external_float64_ptr(cnode,"my_sub",data,5_8)
        ! change the first element in the array
        ! so we can check the external semantics
        data(1) = 3.1415d+0
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! float64 res = n.as_float64()
        res = conduit_node_fetch_path_as_float64(cnode,"my_sub")
        call assert_equals(res,3.1415d+0)
        ! float64 *res_ptr = n.as_float64_ptr()
        call conduit_node_fetch_path_as_float64_ptr(cnode,"my_sub",f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));
        
        call assert_equals(f_arr(1),3.1415d+0)
        
        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_and_fetch_path_external_float64_ptr


!------------------------------------------------------------------------------
end module f_conduit_node_float64
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
integer(C_INT) function fortran_test() bind(C,name="fortran_test")
!------------------------------------------------------------------------------
  use fruit
  use f_conduit_node_float64
  implicit none
  logical res
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_node_set_float32
  call t_node_set_float32_ptr
  call t_node_set_external_float32_ptr

  call t_node_as_float32
  call t_node_as_float32_ptr
  call t_node_as_float32_ptr_read_scalar
 
  call t_node_set_and_fetch_path_float32
  call t_node_set_and_fetch_path_float32_ptr
  call t_node_set_and_fetch_path_external_float32_ptr

  call t_node_set_float64
  call t_node_set_float64_ptr
  call t_node_set_external_float64_ptr

  call t_node_as_float64
  call t_node_as_float64_ptr
  call t_node_as_float64_ptr_read_scalar
 
  call t_node_set_and_fetch_path_float64
  call t_node_set_and_fetch_path_float64_ptr
  call t_node_set_and_fetch_path_external_float64_ptr
    
  
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


