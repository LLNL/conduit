!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.

!------------------------------------------------------------------------------
!
! f_conduit_intro_ftn_example.f
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module f_conduit_intro_ftn_example
!------------------------------------------------------------------------------

  use iso_c_binding
  use fruit
  use conduit
  use conduit_obj
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------    
!  Opaque Pointer Function Style test
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
        subroutine t_ftn_obj_example
            type(node) n, n_den, n_den_units
            real(kind=8), dimension(4) :: den
            real(kind=8), pointer :: d_arr(:)
            character, pointer :: units(:)
            integer i
            integer units_len

            do i = 1,4
                den(i) = 1.0
            enddo

            n = conduit_node_obj_create()
            call n%set_path_ptr("fields/density/values", den, 4_8)
            call n%set_path("fields/density/units", "g/cc")
            
            n_den = n%fetch("fields/density")
            
            call n_den%fetch_path_as_float64_ptr("values", d_arr)
            
            n_den_units = n_den%fetch("units")
            units_len = n_den_units%number_of_elements()
            
            call n_den_units%as_char8_str(units)
            
            call n_den%print()
            
            print *,"Density (", (units(i),i=1,units_len), "):"
            
            do i = 1,4
                write (*,"(f5.2,1x)",advance="no") d_arr(i)
            enddo
            print *
            
            call conduit_node_obj_destroy(n)
        end subroutine t_ftn_obj_example



!------------------------------------------------------------------------------
end module f_conduit_intro_ftn_example
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program fortran_test
!------------------------------------------------------------------------------
  use fruit
  use f_conduit_intro_ftn_example
  implicit none
  logical ok
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_ftn_obj_example

  call fruit_summary
  call fruit_finalize
  call is_all_successful(ok)
  
  if (.not. ok) then
     call exit(1)
  endif

!------------------------------------------------------------------------------
end program fortran_test
!------------------------------------------------------------------------------


