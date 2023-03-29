!* Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Conduit.


!------------------------------------------------------------------------------
! file: conduit_fortran_endianness_types.f90
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
!--- conduit_endianness_type_id is an Enumeration used to describe the
!--- endianness cases supported by conduit
!-------------------------------------------------------------------------------

module conduit_endianness_enum

implicit none

enum, bind(c)

    ! Use this with `integer(kind(conduit_endianness_type_id))` to declare variables of this type.
    enumerator :: conduit_endianness_type_id = 0

    enumerator :: conduit_endianness_default_id  = 0
    enumerator :: conduit_endianness_big_id = 1
    enumerator :: conduit_endianness_little_id = 2

end enum

end module conduit_endianness_enum
