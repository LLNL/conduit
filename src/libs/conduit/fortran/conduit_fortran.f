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
        procedure :: print => conduit_node_print
        procedure :: set_int    => conduit_node_set_int
        procedure :: set_double => conduit_node_set_double
        procedure :: as_int     => conduit_node_as_int
        procedure :: as_double  => conduit_node_as_double
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
    subroutine c_conduit_node_print(cnode) &
        bind(C, name="conduit_node_print")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
    end subroutine c_conduit_node_print
    end interface

contains

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
    subroutine conduit_node_print(obj)
           use iso_c_binding
           implicit none
           class(node) :: obj
           call c_conduit_node_print(obj%cnode)
    end subroutine conduit_node_print

end module conduit

