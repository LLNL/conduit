!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!
!* Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
! t_f_blueprint_smoke.f
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module t_f_blueprint_smoke
!------------------------------------------------------------------------------

  use iso_c_binding
  use fruit
  use conduit
  use conduit_blueprint
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------    
! About test
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_blueprint_about
        type(C_PTR) cnode
        integer res
        
        !----------------------------------------------------------------------
        call set_case_name("t_blueprint_about")
        !----------------------------------------------------------------------
        
        cnode = conduit_node_create()
        call conduit_blueprint_about(cnode)
        call conduit_node_print(cnode)
        call conduit_node_destroy(cnode)
    
    end subroutine t_blueprint_about

!------------------------------------------------------------------------------
end module t_f_blueprint_smoke
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program fortran_test
!------------------------------------------------------------------------------
  use fruit
  use t_f_blueprint_smoke
  implicit none
  logical ok
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_blueprint_about

  call fruit_summary
  call fruit_finalize
  call is_all_successful(ok)

  if (.not. ok) then
     call exit(1)
  endif

!------------------------------------------------------------------------------
end program fortran_test
!------------------------------------------------------------------------------


