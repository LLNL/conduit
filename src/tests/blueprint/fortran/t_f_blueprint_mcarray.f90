!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.

!------------------------------------------------------------------------------
!
! t_f_blueprint_mcarray.f
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module t_f_blueprint_mcarray
!------------------------------------------------------------------------------

  use iso_c_binding
  use fruit
  use conduit
  use conduit_blueprint
  use conduit_blueprint_mcarray
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------    
! About test
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_blueprint_mcarray_create_and_verify
        type(C_PTR) n
        type(C_PTR) nxform
        type(C_PTR) nempty
        type(C_PTR) info
    
        !----------------------------------------------------------------------
        call set_case_name("t_blueprint_mcarray_create_and_verify")
        !----------------------------------------------------------------------
    
        n = conduit_node_create()
        nxform = conduit_node_create()
        nempty = conduit_node_create()
        info = conduit_node_create()
    
        call conduit_blueprint_mcarray_examples_xyz("interleaved",10_8,n)
        call assert_true( conduit_blueprint_mcarray_verify(n,info) .eqv. .true. )
        call assert_true( conduit_blueprint_mcarray_is_interleaved(n) .eqv. .true. );

        call assert_true( conduit_blueprint_mcarray_to_contiguous(n,nxform) .eqv. .true. )
        call assert_true( conduit_blueprint_mcarray_is_interleaved(nxform) .eqv. .false. );
        call assert_true( conduit_node_is_contiguous(nxform) .eqv. .true. )

        call conduit_blueprint_mcarray_examples_xyz("separate",10_8,n)
        call assert_true( conduit_blueprint_mcarray_verify(n,info) .eqv. .true. )
        call assert_true( conduit_blueprint_mcarray_verify(n,info) .eqv. .true. )


        call conduit_blueprint_mcarray_examples_xyz("contiguous",10_8,n)
        call assert_true( conduit_blueprint_mcarray_verify(n,info) .eqv. .true. )
        call assert_true( conduit_node_is_contiguous(n) .eqv. .true. )
        call assert_true( conduit_blueprint_mcarray_is_interleaved(n) .eqv. .false. );


        call assert_true( conduit_blueprint_mcarray_to_interleaved(n,nxform) .eqv. .true. )
        call assert_true( conduit_blueprint_mcarray_is_interleaved(nxform) .eqv. .true. );

        call conduit_blueprint_mcarray_examples_xyz("interleaved_mixed",10_8,n)
        call assert_true( conduit_blueprint_mcarray_verify(n,info) .eqv. .true. )
        call assert_true( conduit_blueprint_mcarray_verify_sub_protocol("sub",nempty,info) .eqv. .false. )

        call assert_true( conduit_blueprint_mcarray_verify(nempty,info) .eqv. .false. )


        call conduit_node_destroy(n);
        call conduit_node_destroy(nxform);
        call conduit_node_destroy(nempty);
        call conduit_node_destroy(info);
    
    end subroutine t_blueprint_mcarray_create_and_verify

!------------------------------------------------------------------------------
end module t_f_blueprint_mcarray
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program fortran_test
!------------------------------------------------------------------------------
  use fruit
  use t_f_blueprint_mcarray
  implicit none
  logical ok
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_blueprint_mcarray_create_and_verify

  call fruit_summary
  call fruit_finalize
  call is_all_successful(ok)

  if (.not. ok) then
     call exit(1)
  endif

!------------------------------------------------------------------------------
end program fortran_test
!------------------------------------------------------------------------------


