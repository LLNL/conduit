!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!
!* Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
!* 
!* Produced at the Lawrence Livermore National Laboratory
!* 
!* LLNL-CODE-666778
!* 
!* All rights reserved.
!* 
!* This file is part of Conduit. 
!* 
!* For details, see: http://software.llnl.gov/conduit/.
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

