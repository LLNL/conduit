!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.

!------------------------------------------------------------------------------
! conduit_blueprint_mesh.f
!------------------------------------------------------------------------------


!------------------------------------------------------------------------------
module conduit_blueprint_mesh
!------------------------------------------------------------------------------
    use, intrinsic :: iso_c_binding, only : C_PTR
    implicit none

    !--------------------------------------------------------------------------
    interface
    !--------------------------------------------------------------------------

        !----------------------------------------------------------------------
        function conduit_blueprint_mesh_verify(cnode,cinfo) result(res) &
                bind(C, name="conduit_blueprint_mesh_verify")
             use iso_c_binding
             implicit none
             type(C_PTR), value, intent(IN) :: cnode
             type(C_PTR), value, intent(IN) :: cinfo
             logical(C_BOOL) ::res
        end function conduit_blueprint_mesh_verify


        !----------------------------------------------------------------------
        function c_conduit_blueprint_mesh_verify_sub_protocol(protocol,cnode,cinfo) result(res) &
                bind(C, name="conduit_blueprint_mesh_verify_sub_protocol")
            use iso_c_binding
            implicit none
            character(kind=C_CHAR), intent(IN) :: protocol(*)
            type(C_PTR), value, intent(IN) :: cnode
            type(C_PTR), value, intent(IN) :: cinfo
            logical(C_BOOL) ::res
        end function c_conduit_blueprint_mesh_verify_sub_protocol

        !----------------------------------------------------------------------
        subroutine c_conduit_blueprint_mesh_generate_index(cmesh,ref_path,num_domains,cindex_dest) &
                bind(C, name="conduit_blueprint_mesh_generate_index")
            use iso_c_binding
            implicit none
            type(C_PTR), value, intent(IN) :: cmesh
            character(kind=C_CHAR), intent(IN) :: ref_path(*)
            integer(C_SIZE_T), value, intent(in) :: num_domains
            type(C_PTR), value, intent(IN) :: cindex_dest
        end subroutine c_conduit_blueprint_mesh_generate_index

        !----------------------------------------------------------------------
        subroutine c_conduit_blueprint_mesh_examples_braid(mesh_type,nx,ny,nz,cdest) &
                bind(C, name="conduit_blueprint_mesh_examples_braid")
            use iso_c_binding
            implicit none
            character(kind=C_CHAR), intent(IN) :: mesh_type(*)
            integer(C_SIZE_T), value, intent(in) :: nx
            integer(C_SIZE_T), value, intent(in) :: ny
            integer(C_SIZE_T), value, intent(in) :: nz
            type(C_PTR), value, intent(IN) :: cdest
        end subroutine c_conduit_blueprint_mesh_examples_braid

     !-------------------------------------------------------------------------
     end interface
     !-------------------------------------------------------------------------

     !-------------------------------------------------------------------------
     !
     contains
     !
     !-------------------------------------------------------------------------

         !---------------------------------------------------------------------
         function conduit_blueprint_mesh_verify_sub_protocol(protocol,cnode,cinfo) result(res)
             use iso_c_binding
             implicit none
             character(*), intent(IN) :: protocol
             type(C_PTR), value, intent(IN) :: cnode
             type(C_PTR), value, intent(IN) :: cinfo
             logical(C_BOOL) :: res
             !---
             res = c_conduit_blueprint_mesh_verify_sub_protocol(trim(protocol) // C_NULL_CHAR, &
                                                                cnode, &
                                                                cinfo)
         end function conduit_blueprint_mesh_verify_sub_protocol


         !---------------------------------------------------------------------
         subroutine conduit_blueprint_mesh_generate_index(cmesh,ref_path,num_domains,cindex_dest)
             use iso_c_binding
             implicit none
             type(C_PTR), value, intent(IN) :: cmesh
             character(*), intent(IN) :: ref_path
             integer(C_SIZE_T), value, intent(in) :: num_domains
             type(C_PTR), value, intent(IN) :: cindex_dest
             !---
             call c_conduit_blueprint_mesh_generate_index(cmesh, &
                                                          trim(ref_path) // C_NULL_CHAR, &
                                                          num_domains, &
                                                          cindex_dest)
         end subroutine conduit_blueprint_mesh_generate_index


         !---------------------------------------------------------------------
         subroutine conduit_blueprint_mesh_examples_braid(mesh_type,nx,ny,nz,cdest)
             use iso_c_binding
             implicit none
             character(*), intent(IN) :: mesh_type
             integer(C_SIZE_T), value, intent(in) :: nx
             integer(C_SIZE_T), value, intent(in) :: ny
             integer(C_SIZE_T), value, intent(in) :: nz
             type(C_PTR), value, intent(IN) :: cdest
             !---
             call c_conduit_blueprint_mesh_examples_braid(trim(mesh_type) // C_NULL_CHAR, &
                                                          nx,ny,nz, &
                                                          cdest)
         end subroutine conduit_blueprint_mesh_examples_braid

!------------------------------------------------------------------------------
end module conduit_blueprint_mesh
!------------------------------------------------------------------------------

