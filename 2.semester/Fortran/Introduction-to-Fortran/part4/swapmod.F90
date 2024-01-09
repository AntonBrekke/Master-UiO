! 
! Written by Ole W. Saastad, UiO
! December 2022.
! 
! Compile gfortran modswap.F90
!
! module load  GCC/12.2.0 to get 2008 standard.
! The 12.2.0 GNU Fortran implements the Fortran 77, 90 and 95 standards completely, 
! most of the Fortran 2003 and 2008 standards, and some features from the 2018 standard. 
!
!
! This is module containing routines to swap the two arguments, they can
! be integers 32/64 bit, real or complex 32/64 bit. 
!

module swp
  interface swap
     module procedure swap_r32, swap_r64, swap_c32, swap_c64, swap_i32, swap_i64
  end interface swap
  
! Here we make use of a macro using the C preprocessor, 
! .F08 trigger invokation of the C-preprocessor.
! It will expand 'SWP(a,b)' to : 'temp=a; a=b; b=temp' 
! test using cpp swap2.F90 | grep temp
!
#define SWP(x,y)  temp=x; x=y; y=temp

contains
  subroutine swap_r32(a,b)
    use iso_fortran_env
    implicit none
    real(real32), intent(inout) :: a, b
    real(kind(a)) :: temp
    SWP(a,b)
  end subroutine swap_r32
  
  subroutine swap_r64(a,b)
    use iso_fortran_env
    implicit none
    real(real64), intent(inout) :: a, b
    real(kind(a)) :: temp
    SWP(a,b)
  end subroutine swap_r64

  subroutine swap_c32(a,b)
    use iso_fortran_env
    implicit none
    complex(real32), intent(inout) :: a, b
    complex(kind(a)) :: temp
    SWP(a,b)
  end subroutine swap_c32
  
  subroutine swap_c64(a,b)
    use iso_fortran_env
    implicit none
    complex(real64), intent(inout) :: a, b
    complex(kind(a)) :: temp
    SWP(a,b)
  end subroutine swap_c64
    
  subroutine swap_i32(a,b)
    use iso_fortran_env
    implicit none
    integer(int32), intent(inout) :: a, b
    integer(kind(a)) :: temp
    temp = a ; a=b ; b=temp
  end subroutine swap_i32
  
  subroutine swap_i64(a,b)
    use iso_fortran_env
    implicit none
    integer(int64), intent(inout) :: a, b
    integer(kind(a)) :: temp
    temp = a ; a=b ; b=temp
  end subroutine swap_i64
  
end module swp

