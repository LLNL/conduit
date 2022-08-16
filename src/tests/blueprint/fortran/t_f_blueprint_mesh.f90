!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.

!------------------------------------------------------------------------------
!
! t_f_blueprint_mesh.f
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module t_f_blueprint_mesh
!------------------------------------------------------------------------------

  use iso_c_binding
  use fruit
  use conduit
  use conduit_blueprint
  use conduit_blueprint_mesh
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------    
! About test
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_blueprint_mesh_create_and_verify
        type(C_PTR) n
        type(C_PTR) nempty
        type(C_PTR) ntopo
        type(C_PTR) nindex
        type(C_PTR) info
    
        !----------------------------------------------------------------------
        call set_case_name("t_blueprint_mesh_create_and_verify")
        !----------------------------------------------------------------------
    
        n = conduit_node_create()
        nempty = conduit_node_create()
        nindex = conduit_node_create()
        info = conduit_node_create()
    
        call assert_true( logical(conduit_blueprint_mesh_verify(nempty,info) .eqv. .true. , "verify true on empty"))
        call conduit_blueprint_mesh_examples_braid("hexs",3_8,3_8,3_8,n)
        call assert_true( logical(conduit_blueprint_mesh_verify(n,info) .eqv. .true., "verify true on braid hexs"))


        ntopo = conduit_node_fetch(n,"topologies/mesh")

        call assert_true( logical(conduit_blueprint_mesh_verify_sub_protocol("topology",ntopo,info) .eqv. .true.))
        call conduit_node_print(info)
        call assert_true( logical(conduit_blueprint_mesh_verify_sub_protocol("coordset",ntopo,info) .eqv. .false.))
        
        call conduit_blueprint_mesh_generate_index(n," ",1_8,nindex)
        call assert_true( logical(conduit_blueprint_mesh_verify_sub_protocol("index",nindex,info) .eqv. .true.))
        

        call conduit_node_destroy(n)
        call conduit_node_destroy(nempty)
        call conduit_node_destroy(nindex)
        call conduit_node_destroy(info)
    
    end subroutine t_blueprint_mesh_create_and_verify

!------------------------------------------------------------------------------
end module t_f_blueprint_mesh
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program fortran_test
!------------------------------------------------------------------------------
  use fruit
  use t_f_blueprint_mesh
  implicit none
  logical ok
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_blueprint_mesh_create_and_verify

  call fruit_summary
  call fruit_finalize
  call is_all_successful(ok)

  if (.not. ok) then
     call exit(1)
  endif

!------------------------------------------------------------------------------
end program fortran_test
!------------------------------------------------------------------------------


