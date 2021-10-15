!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.

!------------------------------------------------------------------------------
! conduit_blueprint_table_fortran.f
!------------------------------------------------------------------------------


!------------------------------------------------------------------------------
module conduit_blueprint_table
!------------------------------------------------------------------------------
    use, intrinsic :: iso_c_binding, only : C_PTR
    implicit none

    !--------------------------------------------------------------------------
    interface
    !--------------------------------------------------------------------------

        !--------------------------------------------------------------------------
        function conduit_blueprint_table_verify(cnode, cinfo) result(res) &
                bind(C, name="conduit_blueprint_table_verify")
            use iso_c_binding
            implicit none
            type(C_PTR), value, intent(IN) :: cnode
            type(C_PTR), value, intent(IN) :: cinfo
            logical(C_BOOL) ::res
        end function conduit_blueprint_table_verify

        !--------------------------------------------------------------------------
        function conduit_blueprint_table_verify_sub_protocol(protocol, cnode, cinfo) result(res) &
                bind(C, name="conduit_blueprint_table_verify_sub_protocol")
            use iso_c_binding
            implicit none
            character(kind=C_CHAR), intent(IN) :: protocol(*)
            type(C_PTR), value, intent(IN) :: cnode
            type(C_PTR), value, intent(IN) :: cinfo
            logical(C_BOOL) ::res
        end function conduit_blueprint_table_verify_sub_protocol

        !--------------------------------------------------------------------------
        subroutine conduit_blueprint_table_examples_basic(nx, ny, nz, cnode) &
                bind(C, name="conduit_blueprint_table_examples_basic")
            use iso_c_binding
            implicit none
            integer(C_SIZE_T), value, intent(in) :: nx
            integer(C_SIZE_T), value, intent(in) :: ny
            integer(C_SIZE_T), value, intent(in) :: nz
            type(C_PTR), value, intent(IN) :: cnode
        end subroutine conduit_blueprint_table_examples_basic

    !--------------------------------------------------------------------------
    end interface
    !--------------------------------------------------------------------------

!------------------------------------------------------------------------------
end module conduit_blueprint_table
!------------------------------------------------------------------------------
