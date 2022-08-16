!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.

!------------------------------------------------------------------------------
!
! f_conduit_node_int32.f
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module f_conduit_node_int32
!------------------------------------------------------------------------------

  use iso_c_binding
  use fruit
  use conduit
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------    
! int32 tests
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_node_set_int32
        type(C_PTR) cnode
        integer(4) res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_int32")
        !----------------------------------------------------------------------

        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        cnode = conduit_node_create()
        ! n.set_int32(42);
        call conduit_node_set_int32(cnode,42)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! int32 res = n.as_int32();
        res = conduit_node_as_int32(cnode)
        call assert_equals (42, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_int32

    !--------------------------------------------------------------------------
    subroutine t_node_set_int32_ptr
        type(C_PTR) cnode
        integer(4), dimension(5) :: data
        integer nele
        integer i
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_int32_ptr")
        !----------------------------------------------------------------------
        
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! Node n.set_int32_ptr(data,5);
        call conduit_node_set_int32_ptr(cnode,data,5_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! index_t nele = n.dtype().number_of_elements();
        nele = conduit_node_number_of_elements(cnode)
        call assert_equals(nele,5)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_int32_ptr

    !--------------------------------------------------------------------------
    subroutine t_node_set_external_int32_ptr
        type(C_PTR) cnode
        integer(4), dimension(5) :: data
        integer res
        integer i
        integer(4), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_external_int32_ptr")
        !----------------------------------------------------------------------
        
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_int32_ptr(data,5);
        call conduit_node_set_external_int32_ptr(cnode,data,5_8)
        call assert_true( logical(conduit_node_is_data_external(cnode) .eqv. .true. ))
        ! change the first element in the array
        ! so we can check the external semantics
        data(1) = 42
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! int32 res = n.as_int32()
        res = conduit_node_as_int32(cnode)
        call assert_equals(res,42)
        ! int32 *res_ptr = n.as_int32_ptr()
        call conduit_node_as_int32_ptr(cnode,f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));
        call assert_equals(f_arr(1),42)
        
        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_external_int32_ptr

    !--------------------------------------------------------------------------
    subroutine t_node_as_int32
        type(C_PTR) cnode
        type(C_PTR) n1
        integer res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_int32")
        !----------------------------------------------------------------------
                
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_int32(42)
        call conduit_node_set_int32(cnode,42)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! int32 res = n.as_int32();
        res = conduit_node_as_int32(cnode)
        call assert_equals (42, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_as_int32

    !--------------------------------------------------------------------------
    subroutine t_node_as_int32_ptr
        type(C_PTR) cnode
        integer(4), dimension(5) :: data
        integer nele
        integer i
        integer(4), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_int32_ptr")
        !----------------------------------------------------------------------
        
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_int32_ptr(data,5);
        call conduit_node_set_int32_ptr(cnode,data,5_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! index_t nele = n.dtype().number_of_elements();
        nele = conduit_node_number_of_elements(cnode)
        call assert_equals(nele,5)
        ! int32 *res_ptr = n.as_int32_ptr()
        call conduit_node_as_int32_ptr(cnode,f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));
        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_as_int32_ptr


    !--------------------------------------------------------------------------
    subroutine t_node_as_int32_ptr_read_scalar
        type(C_PTR) cnode
        integer nele
        integer i
        integer(4), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_int32_ptr_read_scalar")
        !----------------------------------------------------------------------
                
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_int32(42);
        call conduit_node_set_int32(cnode,42)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! index_t nele = n.dtype().number_of_elements();
        nele = conduit_node_number_of_elements(cnode)
        call assert_equals(nele,1)
        ! int32 *ptr = n.as_int32_ptr();
        call conduit_node_as_int32_ptr(cnode,f_arr)
        ! check if ptr[0] == 42
        call assert_equals(f_arr(1),42)

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_as_int32_ptr_read_scalar


    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_int32
        type(C_PTR) cnode
        type(C_PTR) n1
        integer res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_int32")
        !----------------------------------------------------------------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;     
        cnode = conduit_node_create()
        ! n["my_sub"].set_int32(42)
        ! // or
        ! n.set_path_int32("my_sub",42)
        call conduit_node_set_path_int32(cnode,"my_sub",42)
        ! n.print_detailed()
        call conduit_node_print_detailed(cnode)
        ! int32 res = n["my_sub"].as_int32();
        res = conduit_node_fetch_path_as_int32(cnode,"my_sub")
        call assert_equals (42, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_and_fetch_path_int32


    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_int32_ptr
        type(C_PTR) cnode
        integer(4), dimension(5) :: data
        integer res
        integer i
        integer(4), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_int32_ptr")
        !----------------------------------------------------------------------
        
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_int32_ptr(data,5);
        call conduit_node_set_path_int32_ptr(cnode,"my_sub",data,5_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! int32 *res_ptr = n.as_int32_ptr()
        call conduit_node_fetch_path_as_int32_ptr(cnode,"my_sub",f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));

        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_and_fetch_path_int32_ptr


    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_external_int32_ptr
        type(C_PTR) cnode
        integer(4), dimension(5) :: data
        integer res
        integer i
        integer(4), pointer :: f_arr(:)
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_external_int32_ptr")
        !----------------------------------------------------------------------
        
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_int32_ptr(data,5);
        call conduit_node_set_path_external_int32_ptr(cnode,"my_sub",data,5_8)
        ! change the first element in the array
        ! so we can check the external semantics
        data(1) = 42
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! int32 res = n.as_int32()
        res = conduit_node_fetch_path_as_int32(cnode,"my_sub")
        call assert_equals(res,42)
        ! int32 *res_ptr = n.as_int32_ptr()
        call conduit_node_fetch_path_as_int32_ptr(cnode,"my_sub",f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));
        
        call assert_equals(f_arr(1),42)
        
        ! check array value equiv
        do i = 1,5
            call assert_equals(f_arr(i),data(i))
        enddo

        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_and_fetch_path_external_int32_ptr


!------------------------------------------------------------------------------    
! int64 tests
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_node_set_int64
        type(C_PTR) cnode
        integer(8) res
    
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_int64")
        !----------------------------------------------------------------------

        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        cnode = conduit_node_create()
        ! n.set_int64(42_8);
        call conduit_node_set_int64(cnode,42_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! int64 res = n.as_int64();
        res = conduit_node_as_int64(cnode)
        !call assert_equals (42_8, res)
        call conduit_node_destroy(cnode)
    
    end subroutine t_node_set_int64

    !--------------------------------------------------------------------------
    subroutine t_node_set_int64_ptr
        type(C_PTR) cnode
        integer(8), dimension(5) :: data
        integer nele
        integer i
    
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_int64_ptr")
        !----------------------------------------------------------------------
    
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
    
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! Node n.set_int64_ptr(data,5);
        call conduit_node_set_int64_ptr(cnode,data,5_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! index_t nele = n.dtype().number_of_elements();
        nele = conduit_node_number_of_elements(cnode)
        call assert_equals(nele,5)
        call conduit_node_destroy(cnode)
    
    end subroutine t_node_set_int64_ptr

    !--------------------------------------------------------------------------
    subroutine t_node_set_external_int64_ptr
        type(C_PTR) cnode
        integer(8), dimension(5) :: data
        integer(8) res
        integer i
        integer(8), pointer :: f_arr(:)
    
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_external_int64_ptr")
        !----------------------------------------------------------------------
    
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
    
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_int64_ptr(data,5);
        call conduit_node_set_external_int64_ptr(cnode,data,5_8)
        ! change the first element in the array
        ! so we can check the external semantics
        data(1) = 42_8
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! int64 res = n.as_int64()
        res = conduit_node_as_int64(cnode)
        !=======
        ! NOTE:  fruit doesn't support assert_equals with integer(8)
        !=======
        !call assert_equals(res,42_8)
        print *,res, "vs", 42_8
        ! int64 *res_ptr = n.as_int64_ptr()
        call conduit_node_as_int64_ptr(cnode,f_arr)
    
        call assert_equals(size(data),size(f_arr));
        !call assert_equals(f_arr(1),42_8)
    
        ! check array value equiv
        do i = 1,5
            !=======
            ! NOTE:  fruit doesn't support assert_equals with integer(8)
            !=======
            !call assert_equals(f_arr(i),data(i))
            print *,f_arr(i)," vs ", data(i)
        enddo

        call conduit_node_destroy(cnode)
    
    end subroutine t_node_set_external_int64_ptr

    !--------------------------------------------------------------------------
    subroutine t_node_as_int64
        type(C_PTR) cnode
        type(C_PTR) n1
        integer(8) res
    
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_int64")
        !----------------------------------------------------------------------
            
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_int64(42_8)
        call conduit_node_set_int64(cnode,42_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! int64 res = n.as_int64();
        res = conduit_node_as_int64(cnode)
        !call assert_equals (42_8, res)
        call conduit_node_destroy(cnode)
    
    end subroutine t_node_as_int64

    !--------------------------------------------------------------------------
    subroutine t_node_as_int64_ptr
        type(C_PTR) cnode
        integer(8), dimension(5) :: data
        integer nele
        integer i
        integer(8), pointer :: f_arr(:)
    
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_int64_ptr")
        !----------------------------------------------------------------------
    
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
    
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_int64_ptr(data,5);
        call conduit_node_set_int64_ptr(cnode,data,5_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! index_t nele = n.dtype().number_of_elements();
        nele = conduit_node_number_of_elements(cnode)
        call assert_equals(nele,5)
        ! int64 *res_ptr = n.as_int64_ptr()
        call conduit_node_as_int64_ptr(cnode,f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));

        ! check array value equiv
        do i = 1,5
            !=======
            ! NOTE:  fruit doesn't support assert_equals with integer(8)
            !=======
            !call assert_equals(f_arr(i),data(i))
            print *,f_arr(i)," vs ", data(i)
        enddo

        call conduit_node_destroy(cnode)
    
    end subroutine t_node_as_int64_ptr


    !--------------------------------------------------------------------------
    subroutine t_node_as_int64_ptr_read_scalar
        type(C_PTR) cnode
        integer nele
        integer i
        integer(8), pointer :: f_arr(:)
    
        !----------------------------------------------------------------------
        call set_case_name("t_node_as_int64_ptr_read_scalar")
        !----------------------------------------------------------------------
            
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_int64(42_8);
        call conduit_node_set_int64(cnode,42_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! index_t nele = n.dtype().number_of_elements();
        nele = conduit_node_number_of_elements(cnode)
        call assert_equals(nele,1)
        ! int64 *ptr = n.as_int64_ptr();
        call conduit_node_as_int64_ptr(cnode,f_arr)
        ! check if ptr[0] == 42_8
        !call assert_equals(f_arr(1),42_8)

        call conduit_node_destroy(cnode)
    
    end subroutine t_node_as_int64_ptr_read_scalar


    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_int64
        type(C_PTR) cnode
        type(C_PTR) n1
        integer res
    
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_int64")
        !----------------------------------------------------------------------
    
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;     
        cnode = conduit_node_create()
        ! n["my_sub"].set_int64(42_8)
        ! // or
        ! n.set_path_int64("my_sub",42_8)
        call conduit_node_set_path_int64(cnode,"my_sub",42_8)
        ! n.print_detailed()
        call conduit_node_print_detailed(cnode)
        ! int64 res = n["my_sub"].as_int64();
        res = conduit_node_fetch_path_as_int64(cnode,"my_sub")
        !call assert_equals (42_8, res)
        call conduit_node_destroy(cnode)
    
    end subroutine t_node_set_and_fetch_path_int64


    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_int64_ptr
        type(C_PTR) cnode
        integer(8), dimension(5) :: data
        integer(8) res
        integer i
        integer(8), pointer :: f_arr(:)
    
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_int64_ptr")
        !----------------------------------------------------------------------
    
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
    
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_int64_ptr(data,5);
        call conduit_node_set_path_int64_ptr(cnode,"my_sub",data,5_8)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! int64 *res_ptr = n.as_int64_ptr()
        call conduit_node_fetch_path_as_int64_ptr(cnode,"my_sub",f_arr)
        
        call assert_equals(size(data),size(f_arr));
        
        ! check array value equiv
        do i = 1,5
            !=======
            ! NOTE:  fruit doesn't support assert_equals with integer(8)
            !=======
            !call assert_equals(f_arr(i),data(i))
            print *,f_arr(i)," vs ", data(i)
        enddo

        call conduit_node_destroy(cnode)
    
    end subroutine t_node_set_and_fetch_path_int64_ptr


    !--------------------------------------------------------------------------
    subroutine t_node_set_and_fetch_path_external_int64_ptr
        type(C_PTR) cnode
        integer(8), dimension(5) :: data
        integer(8) res
        integer i
        integer(8), pointer :: f_arr(:)
    
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_and_fetch_path_external_int64_ptr")
        !----------------------------------------------------------------------
    
        ! fill our 32-bit x5 integer array
        do i = 1,5
            data(i) = i
        enddo
    
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n; 
        cnode = conduit_node_create()
        ! n.set_external_int64_ptr(data,5);
        call conduit_node_set_path_external_int64_ptr(cnode,"my_sub",data,5_8)
        ! change the first element in the array
        ! so we can check the external semantics
        data(1) = 42_8
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! int64 res = n.as_int64()
        res = conduit_node_fetch_path_as_int64(cnode,"my_sub")
        !=======
        ! NOTE:  fruit doesn't support assert_equals with integer(8)
        !=======
        !call assert_equals(res,42)
        print *, res, "vs", 42
        ! int64 *res_ptr = n.as_int64_ptr()
        call conduit_node_fetch_path_as_int64_ptr(cnode,"my_sub",f_arr)
        ! check size of fetched array
        call assert_equals(size(data),size(f_arr));
    
        !call assert_equals(f_arr(1),42_8)
    
        ! check array value equiv
        do i = 1,5
            !=======
            ! NOTE:  fruit doesn't support assert_equals with integer(8)
            !=======
            !call assert_equals(f_arr(i),data(i))
            print *,f_arr(i)," vs ", data(i)
        enddo

        call conduit_node_destroy(cnode)
    
    end subroutine t_node_set_and_fetch_path_external_int64_ptr



!------------------------------------------------------------------------------
end module f_conduit_node_int32
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program fortran_test
!------------------------------------------------------------------------------
  use fruit
  use f_conduit_node_int32
  implicit none
  logical ok
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_node_set_int32
  call t_node_set_int32_ptr
  call t_node_set_external_int32_ptr

  call t_node_as_int32
  call t_node_as_int32_ptr
  call t_node_as_int32_ptr_read_scalar
 
  call t_node_set_and_fetch_path_int32
  call t_node_set_and_fetch_path_int32_ptr
  call t_node_set_and_fetch_path_external_int32_ptr

  call t_node_set_int64
  call t_node_set_int64_ptr
  call t_node_set_external_int64_ptr

  call t_node_as_int64
  call t_node_as_int64_ptr
  call t_node_as_int64_ptr_read_scalar
 
  call t_node_set_and_fetch_path_int64
  call t_node_set_and_fetch_path_int64_ptr
  call t_node_set_and_fetch_path_external_int64_ptr
    
  
  
  call fruit_summary
  call fruit_finalize
  call is_all_successful(ok)
  
  if (.not. ok) then
     call exit(1)
  endif

!------------------------------------------------------------------------------
end program fortran_test
!------------------------------------------------------------------------------


