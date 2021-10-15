!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.

!------------------------------------------------------------------------------
!
! t_f_blueprint_table.f
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module t_f_blueprint_table
    !------------------------------------------------------------------------------
    use iso_c_binding
    use fruit
    use conduit
    use conduit_blueprint
    use conduit_blueprint_table
    implicit none

    !------------------------------------------------------------------------------
    contains
    !------------------------------------------------------------------------------

    !------------------------------------------------------------------------------
    ! About test
    !------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_blueprint_table_basic_and_verify
            type(C_PTR) n
            type(C_PTR) info
            INTEGER(8) nx, ny, nz

        !----------------------------------------------------------------------
        call set_case_name("t_blueprint_table_basic_and_verify")
        !----------------------------------------------------------------------

        n = conduit_node_create()
        info = conduit_node_create()
        nx = 5
        ny = 4
        nz = 3

        ! Ensure verify() on empty node is false
        call assert_true( conduit_blueprint_table_verify(n, info) .eqv. .false. )

        ! Ensure verify() on example node is true
        call conduit_blueprint_table_examples_basic(nx, ny, nz, n)
        call assert_true( conduit_blueprint_table_verify(n, info) .eqv. .true. )

        ! There are no sub protocols
        call assert_true( conduit_blueprint_table_verify_sub_protocol("subproto", n, info) .eqv. .false. )

        call conduit_node_destroy(n);
        call conduit_node_destroy(info);

    end subroutine t_blueprint_table_basic_and_verify

!------------------------------------------------------------------------------
end module t_f_blueprint_table
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program fortran_test
    !------------------------------------------------------------------------------
    use fruit
    use t_f_blueprint_table
    implicit none
    logical ok

    call init_fruit

    !----------------------------------------------------------------------------
    ! call our test routines
    !----------------------------------------------------------------------------
    call t_blueprint_table_basic_and_verify

    call fruit_summary
    call fruit_finalize
    call is_all_successful(ok)

    if (.not. ok) then
        call exit(1)
    endif

!------------------------------------------------------------------------------
end program fortran_test
!------------------------------------------------------------------------------
