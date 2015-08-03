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
!* For details, see: http://cyrush.github.io/conduit.
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
! conduit_fortran_obj.f
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module conduit_obj
!------------------------------------------------------------------------------
    use conduit
    use, intrinsic :: iso_c_binding, only : C_PTR
    implicit none
    
    !--------------------------------------------------------------------------
    type node
        type(C_PTR) cnode
    contains

        !----------------------------------------------------------------------
        procedure :: fetch  => conduit_node_obj_fetch
        procedure :: append => conduit_node_obj_append
        procedure :: child  => conduit_node_obj_child
        procedure :: number_of_children => conduit_node_obj_number_of_children
        procedure :: number_of_elements => conduit_node_obj_number_of_elements
        !----------------------------------------------------------------------
        procedure :: set_int32     => conduit_node_obj_set_int32        
        procedure :: set_int32_ptr => conduit_node_obj_set_int32_ptr
        procedure :: set_external_int32_ptr => conduit_node_obj_set_external_int32_ptr
        procedure :: as_int32      => conduit_node_obj_as_int
        procedure :: as_int32_ptr  => conduit_node_obj_as_int32_ptr
        !----------------------------------------------------------------------
        procedure :: set_path_int32 => conduit_node_obj_set_path_int32
        procedure :: fetch_path_as_int32 => conduit_node_obj_fetch_path_as_int32
    
        !----------------------------------------------------------------------
        procedure :: set_float64 => conduit_node_obj_set_float64
        procedure :: as_float64  => conduit_node_obj_as_float64
        !----------------------------------------------------------------------
        procedure :: set_int    => conduit_node_obj_set_int
        procedure :: as_int     => conduit_node_obj_as_int
        !----------------------------------------------------------------------
        procedure :: set_double => conduit_node_obj_set_double
        procedure :: as_double  => conduit_node_obj_as_double
        !----------------------------------------------------------------------
        procedure :: print => conduit_node_obj_print
        procedure :: print_detailed => conduit_node_obj_print_detailed

    end type node
    !--------------------------------------------------------------------------

!------------------------------------------------------------------------------
!
contains
!
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    function conduit_node_obj_create() result(obj)
        use iso_c_binding
        implicit none
        type(node) :: obj
        obj%cnode = conduit_node_create()
    end function conduit_node_obj_create

    !--------------------------------------------------------------------------
    subroutine conduit_node_obj_destroy(obj)
        use iso_c_binding
        implicit none
        type(node) :: obj
        call conduit_node_destroy(obj%cnode)
        obj%cnode = C_NULL_PTR
    end subroutine conduit_node_obj_destroy

    !--------------------------------------------------------------------------
    function conduit_node_obj_fetch(obj, path) result(res)
         use iso_c_binding
         implicit none
         class(node) :: obj
         character(*) :: path
         type(node) :: res
         res%cnode = conduit_node_fetch(obj%cnode, trim(path) // C_NULL_CHAR)
     end function conduit_node_obj_fetch

    !--------------------------------------------------------------------------
    function conduit_node_obj_append(obj) result(res)
        use iso_c_binding
        implicit none
        class(node) :: obj
        type(node) :: res
        res%cnode = conduit_node_append(obj%cnode)
    end function conduit_node_obj_append

    !--------------------------------------------------------------------------
    function conduit_node_obj_child(obj, idx) result(res)
        use iso_c_binding
        implicit none
        class(node) :: obj
        integer(C_SIZE_T) :: idx
        type(node) :: res
        res%cnode = conduit_node_child(obj%cnode, idx)
    end function conduit_node_obj_child

    !--------------------------------------------------------------------------
    function conduit_node_obj_number_of_children(obj) result(res)
        use iso_c_binding
        implicit none
        class(node) :: obj
        integer(C_SIZE_T) :: res
        res = conduit_node_number_of_children(obj%cnode)
    end function conduit_node_obj_number_of_children

    !--------------------------------------------------------------------------
    function conduit_node_obj_number_of_elements(obj) result(res)
        use iso_c_binding
        implicit none
        class(node) :: obj
        integer(C_SIZE_T) :: res
        res = conduit_node_number_of_elements(obj%cnode)
    end function conduit_node_obj_number_of_elements

    !--------------------------------------------------------------------------
    subroutine conduit_node_obj_set_int32(obj, val)
        use iso_c_binding
        implicit none
        class(node) :: obj
        integer(4) :: val
        call conduit_node_set_int32(obj%cnode, val)
    end subroutine conduit_node_obj_set_int32

    !--------------------------------------------------------------------------
    subroutine conduit_node_obj_set_int32_ptr(obj, data, num_elements) 
        use iso_c_binding
        implicit none
        class(node) :: obj
        integer(4), intent (out), dimension (*) :: data
        integer(C_SIZE_T) :: num_elements
        call conduit_node_set_int32_ptr(obj%cnode,data,num_elements)
    end subroutine conduit_node_obj_set_int32_ptr

    !--------------------------------------------------------------------------
    subroutine conduit_node_obj_set_external_int32_ptr(obj, data, num_elements) 
        use iso_c_binding
        implicit none
        class(node) :: obj
        integer(4), intent (out), dimension (*) :: data
        integer(C_SIZE_T) :: num_elements
        call conduit_node_set_external_int32_ptr(obj%cnode,data,num_elements)
    end subroutine conduit_node_obj_set_external_int32_ptr

    
    !--------------------------------------------------------------------------
    function conduit_node_obj_as_int32(obj) result(res)
           use iso_c_binding
           implicit none
           class(node) :: obj
           integer(4) :: res
           res = conduit_node_as_int32(obj%cnode)
    end function conduit_node_obj_as_int32
    
    !--------------------------------------------------------------------------
    subroutine conduit_node_obj_as_int32_ptr(obj,f_out)
           use iso_c_binding
           implicit none
           class(node) :: obj
           integer(4), pointer :: f_out(:)
           call conduit_node_as_int32_ptr(obj%cnode,f_out)
    end subroutine conduit_node_obj_as_int32_ptr
    
    !--------------------------------------------------------------------------
    subroutine conduit_node_obj_set_path_int32(obj, path, val)
        use iso_c_binding
        implicit none
        class(node) :: obj
        character(*) :: path
        integer(4) :: val
        call conduit_node_set_path_int32(obj%cnode, trim(path) // C_NULL_CHAR, val)
    end subroutine conduit_node_obj_set_path_int32

    !--------------------------------------------------------------------------
    function conduit_node_obj_fetch_path_as_int32(obj,path) result(res)
           use iso_c_binding
           implicit none
           class(node) :: obj
           character(*) :: path
           integer(4) :: res
           res = conduit_node_fetch_path_as_int32(obj%cnode, trim(path) // C_NULL_CHAR)
    end function conduit_node_obj_fetch_path_as_int32

    !--------------------------------------------------------------------------
    subroutine conduit_node_obj_set_int(obj, val)
        use iso_c_binding
        implicit none
        class(node) :: obj
        integer(C_INT) :: val
        call conduit_node_set_int(obj%cnode, val)
    end subroutine conduit_node_obj_set_int

    !--------------------------------------------------------------------------
    function conduit_node_obj_as_int(obj) result(res)
           use iso_c_binding
           implicit none
           class(node) :: obj
           integer(C_INT) :: res
           res = conduit_node_as_int(obj%cnode)
    end function conduit_node_obj_as_int

    !--------------------------------------------------------------------------
    subroutine conduit_node_obj_set_double(obj, val)
        use iso_c_binding
        implicit none
        class(node) :: obj
        real(C_DOUBLE) :: val
        call conduit_node_set_double(obj%cnode, val)
    end subroutine conduit_node_obj_set_double

    !--------------------------------------------------------------------------
    function conduit_node_obj_as_double(obj) result(res)
           use iso_c_binding
           implicit none
           class(node) :: obj
           real(C_DOUBLE) :: res
           res = conduit_node_as_double(obj%cnode)
    end function conduit_node_obj_as_double

    !--------------------------------------------------------------------------
    subroutine conduit_node_obj_set_float64(obj, val)
        use iso_c_binding
        implicit none
        class(node) :: obj
        real(8) :: val
        call conduit_node_set_double(obj%cnode, val)
    end subroutine conduit_node_obj_set_float64

    !--------------------------------------------------------------------------
    function conduit_node_obj_as_float64(obj) result(res)
           use iso_c_binding
           implicit none
           class(node) :: obj
           real(8) :: res
           res = conduit_node_as_float64(obj%cnode)
    end function conduit_node_obj_as_float64

    !--------------------------------------------------------------------------
    subroutine conduit_node_obj_print(obj)
           use iso_c_binding
           implicit none
           class(node) :: obj
           call conduit_node_print(obj%cnode)
    end subroutine conduit_node_obj_print

    !--------------------------------------------------------------------------
    subroutine conduit_node_obj_print_detailed(obj)
           use iso_c_binding
           implicit none
           class(node) :: obj
           call conduit_node_print_detailed(obj%cnode)
    end subroutine conduit_node_obj_print_detailed

!------------------------------------------------------------------------------
end module conduit_obj
!------------------------------------------------------------------------------


