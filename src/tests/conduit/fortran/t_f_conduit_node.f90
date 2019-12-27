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
! f_conduit_node.f
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module f_conduit_node
!------------------------------------------------------------------------------

  use iso_c_binding
  use fruit
  use conduit
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------    
!  Opaque Pointer Function Style test
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_node_create
        type(C_PTR) cnode
        type(C_PTR) cinfo
        integer res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_create")
        !----------------------------------------------------------------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        ! n.print_detailed();
        cnode = conduit_node_create()
        call assert_true(conduit_node_is_root(cnode) .eqv. .true. )
        call conduit_node_print_detailed(cnode)
        ! Node n_info;
        ! n.info(n_info);
        ! n_info.print();
        cinfo = conduit_node_create()
        call conduit_node_info(cnode,cinfo)
        call conduit_node_print(cinfo)

        call conduit_node_destroy(cnode)
        call conduit_node_destroy(cinfo)
    
    end subroutine t_node_create


    !--------------------------------------------------------------------------
    subroutine t_node_append
        type(C_PTR) cnode
        type(C_PTR) n1
        type(C_PTR) n2
        type(C_PTR) na
        type(C_PTR) nb
        integer(4) res_1
        real(8)    res_2
        integer    nchld
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_append")
        !----------------------------------------------------------------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;  
        cnode = conduit_node_create()
        
        ! Node &n1 = n.append();
        n1 = conduit_node_append(cnode)
        ! Node &n2 = n.append();
        n2 = conduit_node_append(cnode)

        call assert_true( conduit_node_is_root(n2) .eqv. .false.)
        
        ! index_t nchld = n.number_of_children();
        nchld = conduit_node_number_of_children(cnode)
        
        call assert_equals(nchld, 2)
        ! n1.set_int32(42);
        call conduit_node_set_int32(n1,42)
        ! n1.set_float64(3.1415);
        call conduit_node_set_float64(n2,3.1415d+0)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        
        ! Node &na = n[0];
        ! // or
        ! Node &na = n.child(0);
        na  = conduit_node_child(cnode,0_8)
        ! Node &nb = n[1];
        ! // or
        ! Node &nb = n.child(1);
        nb  = conduit_node_child(cnode,1_8)

        !int32 res_1 = n.as_int32();
        res_1 = conduit_node_as_int32(n1)
        !int32 res_2 = n.as_float64();
        res_2 = conduit_node_as_float64(n2)
        
        call assert_equals (42, res_1)
        call assert_equals (3.1415d+0, res_2)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_append


    !--------------------------------------------------------------------------
    subroutine t_node_set_int
        type(C_PTR) cnode
        integer res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_int")
        !----------------------------------------------------------------------

        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;    
        cnode = conduit_node_create()
        ! n.set(42);
        call conduit_node_set_int(cnode,42)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! int res = n.as_int();
        res = conduit_node_as_int(cnode)
        call assert_equals (42, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_int

    !--------------------------------------------------------------------------
    subroutine t_node_set_double
        type(C_PTR) cnode
        real(kind=8) res
        
        !--------------
        call set_case_name("t_node_set_double")
        !--------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        cnode = conduit_node_create()
        ! n.set(3.1415);
        call conduit_node_set_double(cnode,3.1415d+0)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! double res = n.as_double();
        res = conduit_node_as_double(cnode)
        call assert_equals(3.1415d+0, res)
        call conduit_node_destroy(cnode)
        
    end subroutine t_node_set_double

    !--------------------------------------------------------------------------
    subroutine t_node_set_float64
        type(C_PTR) cnode
        real(kind=8) res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_float64")
        !----------------------------------------------------------------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n;
        cnode = conduit_node_create()
        ! n.set_float64(3.1415);
        call conduit_node_set_float64(cnode,3.1415d+0)
        ! n.print_detailed();
        call conduit_node_print_detailed(cnode)
        ! float64 res = n.as_float64();
        res = conduit_node_as_float64(cnode)
        call assert_equals(3.1415d+0, res)
        call conduit_node_destroy(cnode)

    end subroutine t_node_set_float64

    !--------------------------------------------------------------------------
    subroutine t_node_set_node
        type(C_PTR) cnode1
        type(C_PTR) cnode2
        real(kind=8) res
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_set_node")
        !----------------------------------------------------------------------
        cnode1 = conduit_node_create()
        cnode2 = conduit_node_create()
        call conduit_node_set_path_float64(cnode1,"a",3.1415d+0)
        call conduit_node_set_path_node(cnode2,"path/to",cnode1)
        call assert_true( conduit_node_has_path(cnode2,"path/to/a") .eqv. .true.)


        call conduit_node_set_path_external_node(cnode2,"another/path/to",cnode1)

        res = conduit_node_fetch_path_as_float64(cnode2,"path/to/a");
        call assert_equals(3.1415d+0, res)

        res = conduit_node_fetch_path_as_float64(cnode2,"another/path/to/a");
        call assert_equals(3.1415d+0, res)

        call conduit_node_set_path_float64(cnode1,"a",42.0d+0)

        res = conduit_node_fetch_path_as_float64(cnode2,"another/path/to/a");
        call assert_equals(42.0d+0, res)
        
        call conduit_node_print(cnode2)
        call conduit_node_destroy(cnode1)
        call conduit_node_destroy(cnode2)

    end subroutine t_node_set_node


    !--------------------------------------------------------------------------
    subroutine t_node_diff
        type(C_PTR) cnode1
        type(C_PTR) cnode2
        type(C_PTR) cinfo
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_diff")
        !----------------------------------------------------------------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n1;
        ! Node n2;
        ! Node info;
        cnode1 = conduit_node_create()
        cnode2 = conduit_node_create()
        cinfo  = conduit_node_create()


        ! n1["a"].set_float64(3.1415);
        call conduit_node_set_path_float64(cnode1,"a",3.1415d+0)
        ! n1.diff(n2,info,1e-12)
        !! there is a diff
        call assert_true( conduit_node_diff(cnode1,cnode2,cinfo,1d-12) .eqv. .true.)
        ! n2["a"].set_float64(3.1415);
        call conduit_node_set_path_float64(cnode2,"a",3.1415d+0)
        ! n1.diff(n2,info,1e-12)
        !! no diff
        call assert_true( conduit_node_diff(cnode1,cnode2,cinfo,1d-12) .eqv. .false.)
        ! n2["b"].set_float64(3.1415);
        call conduit_node_set_path_float64(cnode2,"b",3.1415d+0)
        ! n1.diff(n2,info,1e-12)
        !! there is a diff
        call assert_true( conduit_node_diff(cnode1,cnode2,cinfo,1d-12) .eqv. .true.)
        
        ! n1.diff(n2,info,1e-12)
        !! but no diff compat
        call assert_true( conduit_node_diff_compatible(cnode1,cnode2,cinfo,1d-12) .eqv. .false.)
        
        call conduit_node_destroy(cnode1)
        call conduit_node_destroy(cnode2)
        call conduit_node_destroy(cinfo)

    end subroutine t_node_diff

    !--------------------------------------------------------------------------
    subroutine t_node_update
        type(C_PTR) cnode1
        type(C_PTR) cnode2
        real(kind=8) val

        !----------------------------------------------------------------------
        call set_case_name("t_node_update")
        !----------------------------------------------------------------------
        
        !--------------
        ! c++ ~equiv:
        !--------------
        ! Node n1;
        ! Node n2;
        cnode1 = conduit_node_create()
        cnode2 = conduit_node_create()

        ! n1["a"].set_float64(3.1415);
        call conduit_node_set_path_float64(cnode1,"a",3.1415d+0)
        ! n2.update(n1)
        call conduit_node_update(cnode2, cnode1)

        call assert_true( conduit_node_has_path(cnode2,"a") .eqv. .true.)

        call conduit_node_set_path_float64(cnode1,"a",42.0d+0)
        call conduit_node_set_path_float64(cnode1,"b",52.0d+0)

        ! n2.update_compatible(n1)
        call conduit_node_update_compatible(cnode2, cnode1)
        ! float64 val = n2["a"].value()
        val = conduit_node_fetch_path_as_float64(cnode2,"a");

        call assert_equals(42.0d+0, val)

        call assert_true( conduit_node_has_path(cnode2,"a") .eqv. .true.)
        call assert_true( conduit_node_has_path(cnode2,"b") .eqv. .false.)


        ! n2.update_external(n1)
        call conduit_node_update_external(cnode2, cnode1)
        ! n2["a"].set(float64(62.0));
        call conduit_node_set_path_float64(cnode1,"a",62d+0)
        ! float64 val = n2["a"].value()
        val = conduit_node_fetch_path_as_float64(cnode2,"a");

        call assert_equals(62.0d+0, val)

        call assert_true( conduit_node_has_path(cnode2,"a") .eqv. .true.)
        call assert_true( conduit_node_has_path(cnode2,"b") .eqv. .true.)

        val = conduit_node_fetch_path_as_float64(cnode2,"b");

        call assert_equals(52.0d+0, val)

        call conduit_node_destroy(cnode1)
        call conduit_node_destroy(cnode2)

    end subroutine t_node_update
    
    
    !--------------------------------------------------------------------------
    subroutine t_node_compact_to
        type(C_PTR) cnode1
        type(C_PTR) cnode2
        real(kind=8) val
        integer    bytes_res

        !----------------------------------------------------------------------
        call set_case_name("t_node_compact_to")
        !----------------------------------------------------------------------
        
        cnode1 = conduit_node_create()
        cnode2 = conduit_node_create()

        call conduit_node_set_path_int32(cnode1,"a",10)
        call conduit_node_set_path_int32(cnode1,"b",20)
        call conduit_node_set_path_float64(cnode1,"c",30d+0)

        bytes_res = conduit_node_total_bytes_allocated(cnode1);
        call assert_equals( bytes_res, 16)

        call assert_true( conduit_node_is_contiguous(cnode1) .eqv. .false.)
        
        call conduit_node_compact_to(cnode1,cnode2);

        call assert_true( conduit_node_is_contiguous(cnode2) .eqv. .true.)

        bytes_res = conduit_node_total_bytes_compact(cnode2);
        call assert_equals( bytes_res, 16)
        bytes_res = conduit_node_total_strided_bytes(cnode2)
        call assert_equals( bytes_res, 16)
        bytes_res = conduit_node_total_bytes_allocated(cnode2)
        call assert_equals( bytes_res, 16)

        call conduit_node_destroy(cnode1)
        call conduit_node_destroy(cnode2)

    end subroutine t_node_compact_to
    
    
    !--------------------------------------------------------------------------
    subroutine t_node_remove
        type(C_PTR) cnode
        real(kind=8) val

        !----------------------------------------------------------------------
        call set_case_name("t_node_remove")
        !----------------------------------------------------------------------

        cnode = conduit_node_create()

        call conduit_node_set_path_float64(cnode,"a",62d+0)
        call assert_true( conduit_node_has_path(cnode,"a") .eqv. .true.)
        call conduit_node_remove_path(cnode,"a")
        call assert_true( conduit_node_has_path(cnode,"a") .eqv. .false.)

        call conduit_node_set_path_float64(cnode,"a",62d+0)
        call assert_true( conduit_node_has_path(cnode,"a") .eqv. .true.)
        ! remove child using idx (still using zero-based idx)
        call conduit_node_remove_child(cnode, 0_8)
        call assert_true( conduit_node_has_path(cnode,"a") .eqv. .false.)


        call conduit_node_set_path_float64(cnode,"a",62d+0)
        call conduit_node_rename_child(cnode,"a","b")
        call assert_true( conduit_node_has_path(cnode,"a") .eqv. .false.)
        call assert_true( conduit_node_has_path(cnode,"b") .eqv. .true.)

        val = conduit_node_fetch_path_as_float64(cnode,"b");

        call assert_equals(62.0d+0, val)

        call conduit_node_destroy(cnode)

    end subroutine t_node_remove


   !--------------------------------------------------------------------------
    subroutine t_node_parse
        type(C_PTR) cnode
        real(kind=8) r_val
        integer      i_val
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_parse")
        !----------------------------------------------------------------------
        cnode = conduit_node_create()

        ! json
        call conduit_node_parse(cnode,"a","{\"a\": 42.0 }","json")
        ! float64 val = n["a"].value()
        r_val = conduit_node_fetch_path_as_float64(cnode,"a")
        call assert_equals(42.0d+0, r_val)

        call conduit_node_parse(cnode,"a","{\"a\": 42 }","json")
        ! int64 val = n["a"].value()
        i_val = conduit_node_fetch_path_as_int64(cnode,"a")
        call assert_equals(42, i_val)

        ! yaml
        call conduit_node_parse(cnode,"a","a: 42.0","yaml")
        ! float64 val = n["a"].value()
        r_val = conduit_node_fetch_path_as_float64(cnode,"a")
        call assert_equals(42.0d+0, r_val)

        call conduit_node_parse(cnode,"a","a: 42","yaml")
        ! int64 val = n["a"].value()
        i_val = conduit_node_fetch_path_as_int64(cnode,"a")
        call assert_equals(42, i_val)

        call conduit_node_print(cnode)
        call conduit_node_destroy(cnode)
    end subroutine t_node_parse

   !--------------------------------------------------------------------------
    subroutine t_node_save_load
        type(C_PTR) cnode1
        type(C_PTR) cnode2
        real(kind=8) r_val
        integer      i_val
        
        !----------------------------------------------------------------------
        call set_case_name("t_node_save_load")
        !----------------------------------------------------------------------
        cnode1 = conduit_node_create()
        cnode2 = conduit_node_create()

        call conduit_node_set_path_float64(cnode1,"a",42d+0)
        call conduit_node_set_path_int64(cnode1,"a",42)

        call conduit_node_print(cnode1)

        conduit_node_save(cnode1,"tout_f_node_save.json","json")
        conduit_node_load(cnode2,"tout_f_node_save.json","json")

        call conduit_node_print(cnode2)

        r_val = conduit_node_fetch_path_as_float64(cnode2,"a");
        i_val = conduit_node_fetch_path_as_float64(cnode2,"b");

        call assert_equals(42.0d+0, r_val)
        call assert_equals(42, i_val)

        conduit_node_save(cnode1,"tout_f_node_save.yaml","yaml")
        conduit_node_load(cnode2,"tout_f_node_save.yaml","yaml")

        call conduit_node_print(cnode2)

        r_val = conduit_node_fetch_path_as_float64(cnode2,"a");
        i_val = conduit_node_fetch_path_as_float64(cnode2,"b");

        call assert_equals(42.0d+0, r_val)
        call assert_equals(42, i_val)

        call conduit_node_destroy(cnode1)
        call conduit_node_destroy(cnode2)
    end subroutine t_node_parse




!------------------------------------------------------------------------------
end module f_conduit_node
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program fortran_test
!------------------------------------------------------------------------------
  use fruit
  use f_conduit_node
  implicit none
  logical ok
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_node_create
  call t_node_append
  call t_node_set_int
  call t_node_set_double
  call t_node_set_float64
  call t_node_set_node
  call t_node_diff
  call t_node_update
  call t_node_compact_to
  call t_node_remove
  call t_node_parse
  call t_node_save_load

  call fruit_summary
  call fruit_finalize
  call is_all_successful(ok)
  
  if (.not. ok) then
     call exit(1)
  endif

!------------------------------------------------------------------------------
end program fortran_test
!------------------------------------------------------------------------------


