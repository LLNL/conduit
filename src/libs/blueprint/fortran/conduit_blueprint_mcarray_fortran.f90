!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.

!------------------------------------------------------------------------------
! conduit_blueprint_mcarray.f
!------------------------------------------------------------------------------


!------------------------------------------------------------------------------
module conduit_blueprint_mcarray
!------------------------------------------------------------------------------
    use, intrinsic :: iso_c_binding, only : C_PTR
    implicit none

    !--------------------------------------------------------------------------
    interface
    !--------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    function conduit_blueprint_mcarray_verify(cnode,cinfo) result(res) &
             bind(C, name="conduit_blueprint_mcarray_verify")
         use iso_c_binding
         implicit none
         type(C_PTR), value, intent(IN) :: cnode
         type(C_PTR), value, intent(IN) :: cinfo
         logical(C_BOOL) ::res
     end function conduit_blueprint_mcarray_verify


     !--------------------------------------------------------------------------
     function c_conduit_blueprint_mcarray_verify_sub_protocol(protocol,cnode,cinfo) result(res) &
              bind(C, name="conduit_blueprint_mcarray_verify_sub_protocol")
          use iso_c_binding
          implicit none
          character(kind=C_CHAR), intent(IN) :: protocol(*)
          type(C_PTR), value, intent(IN) :: cnode
          type(C_PTR), value, intent(IN) :: cinfo
          logical(C_BOOL) ::res
      end function c_conduit_blueprint_mcarray_verify_sub_protocol


     !--------------------------------------------------------------------------
     function conduit_blueprint_mcarray_is_interleaved(cnode) result(res) &
              bind(C, name="conduit_blueprint_mcarray_is_interleaved")
          use iso_c_binding
          implicit none
          type(C_PTR), value, intent(IN) :: cnode
          logical(C_BOOL) ::res
      end function conduit_blueprint_mcarray_is_interleaved

      !--------------------------------------------------------------------------
      function conduit_blueprint_mcarray_to_interleaved(cnode,cdest) result(res) &
               bind(C, name="conduit_blueprint_mcarray_to_interleaved")
           use iso_c_binding
           implicit none
           type(C_PTR), value, intent(IN) :: cnode
           type(C_PTR), value, intent(IN) :: cdest
           logical(C_BOOL) ::res
       end function conduit_blueprint_mcarray_to_interleaved

       !--------------------------------------------------------------------------
       function conduit_blueprint_mcarray_to_contiguous(cnode,cdest) result(res) &
                bind(C, name="conduit_blueprint_mcarray_to_contiguous")
            use iso_c_binding
            implicit none
            type(C_PTR), value, intent(IN) :: cnode
            type(C_PTR), value, intent(IN) :: cdest
            logical(C_BOOL) ::res
        end function conduit_blueprint_mcarray_to_contiguous
        
        !--------------------------------------------------------------------------
        subroutine c_conduit_blueprint_mcarray_examples_xyz(mcarray_type,npts,cdest) &
                 bind(C, name="conduit_blueprint_mcarray_examples_xyz")
             use iso_c_binding
             implicit none
             character(kind=C_CHAR), intent(IN) :: mcarray_type(*)
             integer(C_SIZE_T), value, intent(in) :: npts
             type(C_PTR), value, intent(IN) :: cdest
         end subroutine c_conduit_blueprint_mcarray_examples_xyz

     !--------------------------------------------------------------------------
     end interface
     !--------------------------------------------------------------------------


     !------------------------------------------------------------------------------
     !
     contains
     !
     !------------------------------------------------------------------------------

         !--------------------------------------------------------------------------
         function conduit_blueprint_mcarray_verify_sub_protocol(protocol,cnode,cinfo) result(res)
             use iso_c_binding
             implicit none
             character(*), intent(IN) :: protocol
             type(C_PTR), value, intent(IN) :: cnode
             type(C_PTR), value, intent(IN) :: cinfo
             logical(C_BOOL) :: res
             !---
             res = c_conduit_blueprint_mcarray_verify_sub_protocol( trim(protocol) // C_NULL_CHAR, cnode, cinfo)
         end function conduit_blueprint_mcarray_verify_sub_protocol


         !--------------------------------------------------------------------------
         subroutine conduit_blueprint_mcarray_examples_xyz(mcarray_type,npts,cdest)
             use iso_c_binding
             implicit none
             character(*), intent(IN) :: mcarray_type
             integer(C_SIZE_T), value, intent(in) :: npts
             type(C_PTR), value, intent(IN) :: cdest
             !---
             call c_conduit_blueprint_mcarray_examples_xyz(trim(mcarray_type) // C_NULL_CHAR, npts, cdest)
         end subroutine conduit_blueprint_mcarray_examples_xyz

!------------------------------------------------------------------------------
end module conduit_blueprint_mcarray
!------------------------------------------------------------------------------

