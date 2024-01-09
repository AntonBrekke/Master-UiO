!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!
! Example of generic swap routine in fortran 
!
! This makes generic functions swap as opposed to the old standard
! with sswap (single precision), dswap (double), cswap (single complex)
! and zswap for double precision complex variables.
! Not to mention derived data types.
!
! It also show usage of C preprocessor macro.
! 
! Written by Ole W. Saastad, UiO
! December 2022.
! 
! Compile gfortran swapmod.F90 swap.F08
!
! module load  GCC/12.2.0 to get 2008 standard.
! The 12.2.0 GNU Fortran implements the Fortran 77, 90 and 95 standards completely, 
! most of the Fortran 2003 and 2008 standards, and some features from the 2018 standard. 
!
! Any fortran program ending with .F90, F95, F03 & F08 and  will trigger the 
! C-preprocessor and expand macros.
!
! if file have suffix 'f' the option -cpp is needed to trigger cpp.
!


program swapdemo
  use iso_fortran_env
  use swp
  implicit none

  integer(int32) :: a=1_int32, b=2_int32
  integer(int64) :: c=3_int64, d=4_int64
  real(real32)   :: x=1.1_real32, y=2.1_real32
  real(real64)   :: z=3.4_real64, w=4.4_real64
  complex(real64):: u=(6.6_real64, 7.7_real64), v=(8.8_real64, 9.9_real64)
  
  character(len=60) :: fmt ! Check : print *,len(trim(fmt)) of actual string.

  fmt='(2(2(xi2)" ;"),2x2(2(f4.1)" ;"),2x2(2xf3.1","f3.1))'
  write(*,fmt) a, b, c, d, x, y, z, w, u, v

! Same swap routine regardless of data type we provide as parameters.
! The compiler examine the parameters and decide which subroutine to use.

  call swap(a,b)
  call swap(c,d)
  call swap(x,y)
  call swap(z,w)
  call swap(u,v)
  write(*,fmt) a, b, c, d, x, y, z, w, u, v

end program swapdemo
  
