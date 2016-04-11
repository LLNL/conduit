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
! conduit_fortran.f
!------------------------------------------------------------------------------

#include "Bitwidth_Style_Types.inc"

!------------------------------------------------------------------------------
module conduit
!------------------------------------------------------------------------------
    use, intrinsic :: iso_c_binding, only : C_PTR
    implicit none

    !--------------------------------------------------------------------------
    !--------------------------------------------------------------------------
    ! Conduit DataType IDs
    !--------------------------------------------------------------------------
    !--------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    ! generic types
    !--------------------------------------------------------------------------
    integer, parameter :: CONDUIT_EMPTY_ID  = F_CONDUIT_EMPTY_ID
    integer, parameter :: CONDUIT_OBJECT_ID = F_CONDUIT_EMPTY_ID
    integer, parameter :: CONDUIT_LIST_ID   = F_CONDUIT_LIST_ID
    
    !--------------------------------------------------------------------------
    ! bitwidth style signed integer types
    !--------------------------------------------------------------------------
    integer, parameter :: CONDUIT_INT8_ID   = F_CONDUIT_INT8_ID
    integer, parameter :: CONDUIT_INT16_ID  = F_CONDUIT_INT16_ID
    integer, parameter :: CONDUIT_INT32_ID  = F_CONDUIT_INT32_ID
    integer, parameter :: CONDUIT_INT64_ID  = F_CONDUIT_INT64_ID
    !--------------------------------------------------------------------------
    ! bitwidth style unsigned integer types
    !--------------------------------------------------------------------------
    integer, parameter :: CONDUIT_UINT8_ID   = F_CONDUIT_UINT8_ID
    integer, parameter :: CONDUIT_UINT16_ID  = F_CONDUIT_UINT16_ID
    integer, parameter :: CONDUIT_UINT32_ID  = F_CONDUIT_UINT32_ID
    integer, parameter :: CONDUIT_UINT64_ID  = F_CONDUIT_UINT64_ID
    !--------------------------------------------------------------------------
    ! bitwidth style floating point integer types
    !--------------------------------------------------------------------------
    integer, parameter :: CONDUIT_FLOAT32_ID  = F_CONDUIT_FLOAT32_ID
    integer, parameter :: CONDUIT_FLOAT64_ID  = F_CONDUIT_FLOAT64_ID
    !--------------------------------------------------------------------------
    ! string  types 
    !--------------------------------------------------------------------------
    integer, parameter :: CONDUIT_CHAR8_STR_ID = F_CONDUIT_CHAR8_STR_ID

    !--------------------------------------------------------------------------
    ! c style type ids
    !--------------------------------------------------------------------------
    ! these are mapped at configure time to the proper bitwidth style type ids.
    !--------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    ! c style signed integer types
    !--------------------------------------------------------------------------
    integer, parameter :: CONDUIT_CHAR_ID  = F_CONDUIT_NATIVE_CHAR_ID
    integer, parameter :: CONDUIT_SHORT_ID = F_CONDUIT_NATIVE_SHORT_ID
    integer, parameter :: CONDUIT_INT_ID   = F_CONDUIT_NATIVE_INT_ID
    integer, parameter :: CONDUIT_LONG_ID  = F_CONDUIT_NATIVE_LONG_ID
    !--------------------------------------------------------------------------
    ! c style unsigned integer types
    !--------------------------------------------------------------------------
    integer, parameter :: CONDUIT_UCHAR_ID  = F_CONDUIT_NATIVE_UNSIGNED_CHAR_ID
    integer, parameter :: CONDUIT_USHORT_ID = F_CONDUIT_NATIVE_UNSIGNED_SHORT_ID
    integer, parameter :: CONDUIT_UINT_ID   = F_CONDUIT_NATIVE_UNSIGNED_INT_ID
    integer, parameter :: CONDUIT_ULONG_ID  = F_CONDUIT_NATIVE_UNSIGNED_LONG_ID
    !--------------------------------------------------------------------------
    ! c style floating point integer types
    !--------------------------------------------------------------------------
    integer, parameter :: CONDUIT_FLOAT_ID  = F_CONDUIT_NATIVE_FLOAT_ID
    integer, parameter :: CONDUIT_DOUBLE_ID = F_CONDUIT_NATIVE_DOUBLE_ID
    !--------------------------------------------------------------------------



    !--------------------------------------------------------------------------
    interface
    !--------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    function conduit_node_create() result(cnode) &
             bind(C, name="conduit_node_create")
         use iso_c_binding
         implicit none
         type(C_PTR) :: cnode
     end function conduit_node_create

    !--------------------------------------------------------------------------
    subroutine conduit_node_destroy(cnode) &
            bind(C, name="conduit_node_destroy")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
    end subroutine conduit_node_destroy

    !--------------------------------------------------------------------------
    function conduit_node_fetch(cnode, path) result(res) &
             bind(C, name="conduit_node_fetch")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: cnode
         character(kind=C_CHAR), intent(IN) :: path(*)
         type(C_PTR) :: res
     end function conduit_node_fetch

     !--------------------------------------------------------------------------
     function conduit_node_append(cnode) result(res) &
              bind(C, name="conduit_node_append")
          use iso_c_binding
          implicit none
          type(C_PTR), value, intent(IN) :: cnode
          type(C_PTR) :: res
      end function conduit_node_append

      !--------------------------------------------------------------------------
      function conduit_node_child(cnode,idx) result(res) &
               bind(C, name="conduit_node_child")
           use iso_c_binding
           implicit none
           type(C_PTR), value, intent(IN) :: cnode
           integer(C_SIZE_T), value, intent(in) :: idx
           type(C_PTR) :: res
       end function conduit_node_child

    !--------------------------------------------------------------------------
    function conduit_node_number_of_elements(cnode) result(res) &
            bind(C, name="conduit_node_number_of_elements")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        integer(C_SIZE_T) :: res
    end function conduit_node_number_of_elements

    !--------------------------------------------------------------------------
    function conduit_node_number_of_children(cnode) result(res) &
             bind(C, name="conduit_node_number_of_children")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: cnode
         integer(C_SIZE_T) :: res
     end function conduit_node_number_of_children

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_int32(cnode, val) &
                   bind(C, name="conduit_node_set_int32")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        integer(4), value, intent(IN) :: val
    end subroutine conduit_node_set_int32

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_int32_ptr(cnode, data, num_elements) &
                   bind(C, name="conduit_node_set_int32_ptr")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        integer(4), intent (IN), dimension (*) :: data
        integer(C_SIZE_T), value, intent(in) :: num_elements
    end subroutine conduit_node_set_int32_ptr

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_external_int32_ptr(cnode, data, num_elements) &
                   bind(C, name="conduit_node_set_external_int32_ptr")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        integer(4), intent (IN), dimension (*) :: data
        integer(C_SIZE_T), value, intent(in) :: num_elements
    end subroutine conduit_node_set_external_int32_ptr
    
    !--------------------------------------------------------------------------
    subroutine c_conduit_node_set_path_external_float64_ptr(cnode, path, data, num_elements) &
                   bind(C, name="conduit_node_set_path_external_float64_ptr")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(kind=C_CHAR), intent(IN) :: path(*)
        real(8), intent (IN), dimension (*) :: data
        integer(C_SIZE_T), value, intent(in) :: num_elements
    end subroutine c_conduit_node_set_path_external_float64_ptr
    
    !--------------------------------------------------------------------------
    subroutine c_conduit_node_set_path_int32(cnode, path, val) &
                   bind(C, name="conduit_node_set_path_int32")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(kind=C_CHAR), intent(IN) :: path(*)
        integer(4), value, intent(IN) :: val
    end subroutine c_conduit_node_set_path_int32

    !--------------------------------------------------------------------------
    pure function c_conduit_node_fetch_path_as_int32(cnode, path) result(res) &
                   bind(C, name="conduit_node_fetch_path_as_int32")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(kind=C_CHAR), intent(IN) :: path(*)
        integer(4) :: res
    end function c_conduit_node_fetch_path_as_int32

    !--------------------------------------------------------------------------
    pure function conduit_node_as_int32(cnode) result(res) &
             bind(C, name="conduit_node_as_int32")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: cnode
         integer(4) :: res
    end function conduit_node_as_int32
    
    !--------------------------------------------------------------------------
    subroutine conduit_node_set_int(cnode, val) &
                   bind(C, name="conduit_node_set_int")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        integer(C_INT), value, intent(IN) :: val
    end subroutine conduit_node_set_int

    !--------------------------------------------------------------------------
    pure function conduit_node_as_int(cnode) result(res) &
             bind(C, name="conduit_node_as_int")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: cnode
         integer(C_INT) :: res
    end function conduit_node_as_int

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_double(cnode, val) &
                   bind(C, name="conduit_node_set_double")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        real(C_DOUBLE), value, intent(IN) :: val
    end subroutine conduit_node_set_double
    
    !--------------------------------------------------------------------------
    pure function conduit_node_as_double(cnode) result(res) &
             bind(C, name="conduit_node_as_double")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: cnode
         real(C_DOUBLE) :: res
    end function conduit_node_as_double

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_float64(cnode, val) &
                   bind(C, name="conduit_node_set_float64")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        real(8), value, intent(IN) :: val
    end subroutine conduit_node_set_float64
    
    !--------------------------------------------------------------------------
    pure function conduit_node_as_float64(cnode) result(res) &
             bind(C, name="conduit_node_as_float64")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: cnode
         real(8) :: res
    end function conduit_node_as_float64

    !--------------------------------------------------------------------------
    subroutine c_conduit_node_set_path_float64(cnode, path, val) &
                   bind(C, name="conduit_node_set_path_float64")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(kind=C_CHAR), intent(IN) :: path(*)
        real(8), value, intent(IN) :: val
    end subroutine c_conduit_node_set_path_float64

    !--------------------------------------------------------------------------
    pure function conduit_node_fetch_path_as_float64(cnode, path) result(res) &
                   bind(C, name="conduit_node_fetch_path_as_float64")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(kind=C_CHAR), intent(IN) :: path(*)
        real(8) :: res
    end function conduit_node_fetch_path_as_float64

    !--------------------------------------------------------------------------
    subroutine c_conduit_node_set_path_char8_str(cnode, path, val) &
                   bind(C, name="conduit_node_set_path_char8_str")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(kind=C_CHAR), intent(IN) :: path(*)
        character(kind=C_CHAR), intent(IN) :: val(*)
    end subroutine c_conduit_node_set_path_char8_str

    !--------------------------------------------------------------------------
    pure function conduit_node_fetch_path_as_char8_str(cnode, path) result(res) &
                   bind(C, name="conduit_node_fetch_path_as_char8_str")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(kind=C_CHAR), intent(IN) :: path(*)
        character(kind=C_CHAR) :: res
    end function conduit_node_fetch_path_as_char8_str

    !--------------------------------------------------------------------------
    subroutine conduit_node_print(cnode) &
        bind(C, name="conduit_node_print")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
    end subroutine conduit_node_print

    !--------------------------------------------------------------------------
    subroutine conduit_node_print_detailed(cnode) &
        bind(C, name="conduit_node_print_detailed")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
    end subroutine conduit_node_print_detailed

    !--------------------------------------------------------------------------
    function c_conduit_node_as_int32_ptr(cnode) result(int32_ptr) &
             bind(C, name="conduit_node_as_int32_ptr")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: cnode
         type(C_PTR) :: int32_ptr
     end function c_conduit_node_as_int32_ptr

    !--------------------------------------------------------------------------
    end interface
    !--------------------------------------------------------------------------

!------------------------------------------------------------------------------
!
contains
!
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_path_external_float64_ptr(cnode, path, data, num_elements)
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(*), intent(IN) :: path
        real(8), intent (IN), dimension (*) :: data
        integer(C_SIZE_T), value, intent(in) :: num_elements
        call c_conduit_node_set_path_external_float64_ptr(cnode, trim(path) // C_NULL_CHAR, data, num_elements)
    end subroutine conduit_node_set_path_external_float64_ptr
    
    !--------------------------------------------------------------------------
    subroutine conduit_node_set_path_int32(cnode, path, val)
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(*), intent(IN) :: path
        integer(4), value, intent(IN) :: val
        call c_conduit_node_set_path_int32(cnode, trim(path) // C_NULL_CHAR, val)
    end subroutine conduit_node_set_path_int32

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_path_float64(cnode, path, val)
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(*), intent(IN) :: path
        real(8), value, intent(IN) :: val
        call c_conduit_node_set_path_float64(cnode, trim(path) // C_NULL_CHAR, val)
    end subroutine conduit_node_set_path_float64

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_path_char8_str(cnode, path, val)
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(*), intent(IN) :: path
        character(*), intent(IN) :: val
        call c_conduit_node_set_path_char8_str(cnode, trim(path) // C_NULL_CHAR, trim(val) // C_NULL_CHAR)
    end subroutine conduit_node_set_path_char8_str

    !--------------------------------------------------------------------------
    subroutine conduit_node_as_int32_ptr(cnode,f_out)
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        integer(4), pointer :: f_out(:)
        integer n
        type(C_PTR) :: int32_c_ptr
        !---
        n = conduit_node_number_of_elements(cnode)
        int32_c_ptr = c_conduit_node_as_int32_ptr(cnode)
        call c_f_pointer(int32_c_ptr, f_out, (/n/))
    end subroutine conduit_node_as_int32_ptr

    !--------------------------------------------------------------------------
    function conduit_node_fetch_path_as_int32(cnode, path) result(res)
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(*), intent(IN) :: path
        integer(4) :: res
        !---
        res =  c_conduit_node_fetch_path_as_int32(cnode, trim(path) // C_NULL_CHAR)
    end function conduit_node_fetch_path_as_int32



!------------------------------------------------------------------------------
end module conduit
!------------------------------------------------------------------------------

