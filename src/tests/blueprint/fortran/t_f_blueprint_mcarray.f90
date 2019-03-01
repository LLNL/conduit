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


