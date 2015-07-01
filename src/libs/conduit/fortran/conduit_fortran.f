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
! conduit_fortran.f
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module conduit
!------------------------------------------------------------------------------
    use, intrinsic :: iso_c_binding, only : C_PTR
    implicit none

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
        integer(4), intent (out), dimension (*) :: data
        integer(C_SIZE_T), value, intent(in) :: num_elements
    end subroutine conduit_node_set_int32_ptr

    !--------------------------------------------------------------------------
    subroutine conduit_node_set_path_int32(cnode, path, val) &
                   bind(C, name="conduit_node_set_path_int32")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(kind=C_CHAR), intent(IN) :: path(*)
        integer(4), value, intent(IN) :: val
    end subroutine conduit_node_set_path_int32

    !--------------------------------------------------------------------------
    pure function conduit_node_fetch_path_as_int32(cnode, path) result(res) &
                   bind(C, name="conduit_node_fetch_path_as_int32")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) :: cnode
        character(kind=C_CHAR), intent(IN) :: path(*)
        integer(4) :: res
    end function conduit_node_fetch_path_as_int32

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
    end interface
    !--------------------------------------------------------------------------

!------------------------------------------------------------------------------
end module conduit
!------------------------------------------------------------------------------

