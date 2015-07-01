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
! conduit_fortran.f
!------------------------------------------------------------------------------

module conduit
    use, intrinsic :: iso_c_binding, only : C_PTR
    implicit none
    
    type node
        type(C_PTR) cnode
    contains

        !----------------------------------------------------------------------
        procedure :: fetch  => conduit_node_fetch
        procedure :: append => conduit_node_append
        procedure :: child  => conduit_node_child
        procedure :: number_of_children => conduit_node_number_of_children
        !----------------------------------------------------------------------
        procedure :: set_int32_ptr => conduit_node_set_int32_ptr
        procedure :: set_int32  => conduit_node_set_int32
        procedure :: as_int32   => conduit_node_as_int
        !----------------------------------------------------------------------
        procedure :: set_float64 => conduit_node_set_float64
        procedure :: as_float64  => conduit_node_as_float64
        !----------------------------------------------------------------------
        procedure :: set_int    => conduit_node_set_int
        procedure :: as_int     => conduit_node_as_int
        !----------------------------------------------------------------------
        procedure :: set_double => conduit_node_set_double
        procedure :: as_double  => conduit_node_as_double
        !----------------------------------------------------------------------
        procedure :: print => conduit_node_print
        procedure :: print_detailed => conduit_node_print_detailed
    end type node


    interface

    !--------------------------------------------------------------------------
    function c_conduit_node_create() result(obj) &
             bind(C, name="conduit_node_create")
         use iso_c_binding
         implicit none
         type(C_PTR) :: obj
     end function c_conduit_node_create

    !--------------------------------------------------------------------------
    subroutine c_conduit_node_destroy(obj) &
            bind(C, name="conduit_node_destroy")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: obj
    end subroutine c_conduit_node_destroy

    !--------------------------------------------------------------------------
    function c_conduit_node_fetch(obj, path) result(res) &
             bind(C, name="conduit_node_fetch")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: obj
         character(kind=C_CHAR), intent(IN) :: path(*)
         type(C_PTR) :: res
     end function c_conduit_node_fetch

     !--------------------------------------------------------------------------
     function c_conduit_node_append(obj) result(res) &
              bind(C, name="conduit_node_append")
          use iso_c_binding
          implicit none
          type(C_PTR), value, intent(IN) :: obj
          type(C_PTR) :: res
      end function c_conduit_node_append

      !--------------------------------------------------------------------------
      function c_conduit_node_child(obj,idx) result(res) &
               bind(C, name="conduit_node_child")
           use iso_c_binding
           implicit none
           type(C_PTR), value, intent(IN) :: obj
           integer(C_SIZE_T), value, intent(in) :: idx
           type(C_PTR) :: res
       end function c_conduit_node_child

       !--------------------------------------------------------------------------
       function c_conduit_node_number_of_children(obj) result(res) &
                bind(C, name="conduit_node_number_of_children")
            use iso_c_binding
            implicit none
            type(C_PTR), value, intent(IN) :: obj
            integer(C_SIZE_T) :: res
        end function c_conduit_node_number_of_children

    !--------------------------------------------------------------------------
    subroutine c_conduit_node_set_int32(obj, val) &
                   bind(C, name="conduit_node_set_int32")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: obj
        integer(4), value, intent(IN) :: val
    end subroutine c_conduit_node_set_int32

    !--------------------------------------------------------------------------
    subroutine c_conduit_node_set_int32_ptr(obj, data, num_elements) &
                   bind(C, name="conduit_node_set_int32_ptr")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: obj
        integer(4), intent (out), dimension (*) :: data
        integer(C_SIZE_T), value, intent(in) :: num_elements
    end subroutine c_conduit_node_set_int32_ptr

    !--------------------------------------------------------------------------
    pure function c_conduit_node_as_int32(self) result(res) &
             bind(C, name="conduit_node_as_int32")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: self
         integer(4) :: res
    end function c_conduit_node_as_int32
    
    !--------------------------------------------------------------------------
    subroutine c_conduit_node_set_int(obj, val) &
                   bind(C, name="conduit_node_set_int")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: obj
        integer(C_INT), value, intent(IN) :: val
    end subroutine c_conduit_node_set_int

    !--------------------------------------------------------------------------
    pure function c_conduit_node_as_int(self) result(res) &
             bind(C, name="conduit_node_as_int")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: self
         integer(C_INT) :: res
    end function c_conduit_node_as_int

    !--------------------------------------------------------------------------
    subroutine c_conduit_node_set_double(obj, val) &
                   bind(C, name="conduit_node_set_double")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: obj
        real(C_DOUBLE), value, intent(IN) :: val
    end subroutine c_conduit_node_set_double
    
    !--------------------------------------------------------------------------
    pure function c_conduit_node_as_double(self) result(res) &
             bind(C, name="conduit_node_as_double")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: self
         real(C_DOUBLE) :: res
    end function c_conduit_node_as_double

    !--------------------------------------------------------------------------
    subroutine c_conduit_node_set_float64(obj, val) &
                   bind(C, name="conduit_node_set_float64")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: obj
        real(8), value, intent(IN) :: val
    end subroutine c_conduit_node_set_float64
    
    !--------------------------------------------------------------------------
    pure function c_conduit_node_as_float64(self) result(res) &
             bind(C, name="conduit_node_as_float64")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: self
         real(8) :: res
    end function c_conduit_node_as_float64

    !--------------------------------------------------------------------------
    subroutine c_conduit_node_print(cnode) &
        bind(C, name="conduit_node_print")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
    end subroutine c_conduit_node_print

    !--------------------------------------------------------------------------
    subroutine c_conduit_node_print_detailed(cnode) &
        bind(C, name="conduit_node_print_detailed")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
    end subroutine c_conduit_node_print_detailed
    end interface

!------------------------------------------------------------------------------
!
contains
!
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    function conduit_node_create() result(obj)
        use iso_c_binding
        implicit none
        type(node) :: obj
        obj%cnode = c_conduit_node_create()
    end function conduit_node_create

    !--------------------------------------------------------------------------
    subroutine conduit_node_destroy(obj)
        use iso_c_binding
        implicit none
        type(node) :: obj
        call c_conduit_node_destroy(obj%cnode)
        obj%cnode = C_NULL_PTR
    end subroutine conduit_node_destroy

    !--------------------------------------------------------------------------
    function conduit_node_fetch(obj, path) result(res)
         use iso_c_binding
         implicit none
         class(node) :: obj
         character(*) :: path
         type(node) :: res
         res%cnode = c_conduit_node_fetch(obj%cnode, trim(path) // C_NULL_CHAR)
     end function conduit_node_fetch

    !--------------------------------------------------------------------------
    function conduit_node_append(obj) result(res)
        use iso_c_binding
        implicit none
        class(node) :: obj
        type(node) :: res
        res%cnode = c_conduit_node_append(obj%cnode)
    end function conduit_node_append

    !--------------------------------------------------------------------------
    function conduit_node_child(obj, idx) result(res)
        use iso_c_binding
        implicit none
        class(node) :: obj
        integer(C_SIZE_T) :: idx
        type(node) :: res
        res%cnode = c_conduit_node_child(obj%cnode, idx)
    end function conduit_node_child

    !--------------------------------------------------------------------------
    function conduit_node_number_of_children(obj) result(res)
        use iso_c_binding
        implicit none
        class(node) :: obj
        integer(C_SIZE_T) :: res
        res = c_conduit_node_number_of_children(obj%cnode)
    end function conduit_node_number_of_children

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_int32(obj, val)
        use iso_c_binding
        implicit none
        class(node) :: obj
        integer(4) :: val
        call c_conduit_node_set_int32(obj%cnode, val)
    end subroutine conduit_node_set_int32

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_int32_ptr(obj, data, num_elements) 
        use iso_c_binding
        implicit none
        class(node) :: obj
        integer(4), intent (out), dimension (*) :: data
        integer(C_SIZE_T) :: num_elements
        call c_conduit_node_set_int32_ptr(obj%cnode,data,num_elements)
    end subroutine conduit_node_set_int32_ptr

    !--------------------------------------------------------------------------
    function conduit_node_as_int32(obj) result(res)
           use iso_c_binding
           implicit none
           class(node) :: obj
           integer(4) :: res
           res = c_conduit_node_as_int32(obj%cnode)
    end function conduit_node_as_int32

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_int(obj, val)
        use iso_c_binding
        implicit none
        class(node) :: obj
        integer(C_INT) :: val
        call c_conduit_node_set_int(obj%cnode, val)
    end subroutine conduit_node_set_int

    !--------------------------------------------------------------------------
    function conduit_node_as_int(obj) result(res)
           use iso_c_binding
           implicit none
           class(node) :: obj
           integer(C_INT) :: res
           res = c_conduit_node_as_int(obj%cnode)
    end function conduit_node_as_int

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_double(obj, val)
        use iso_c_binding
        implicit none
        class(node) :: obj
        real(C_DOUBLE) :: val
        call c_conduit_node_set_double(obj%cnode, val)
    end subroutine conduit_node_set_double

    !--------------------------------------------------------------------------
    function conduit_node_as_double(obj) result(res)
           use iso_c_binding
           implicit none
           class(node) :: obj
           real(C_DOUBLE) :: res
           res = c_conduit_node_as_double(obj%cnode)
    end function conduit_node_as_double

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_float64(obj, val)
        use iso_c_binding
        implicit none
        class(node) :: obj
        real(8) :: val
        call c_conduit_node_set_double(obj%cnode, val)
    end subroutine conduit_node_set_float64

    !--------------------------------------------------------------------------
    function conduit_node_as_float64(obj) result(res)
           use iso_c_binding
           implicit none
           class(node) :: obj
           real(8) :: res
           res = c_conduit_node_as_float64(obj%cnode)
    end function conduit_node_as_float64

    !--------------------------------------------------------------------------
    subroutine conduit_node_print(obj)
           use iso_c_binding
           implicit none
           class(node) :: obj
           call c_conduit_node_print(obj%cnode)
    end subroutine conduit_node_print

    !--------------------------------------------------------------------------
    subroutine conduit_node_print_detailed(obj)
           use iso_c_binding
           implicit none
           class(node) :: obj
           call c_conduit_node_print_detailed(obj%cnode)
    end subroutine conduit_node_print_detailed

end module conduit

