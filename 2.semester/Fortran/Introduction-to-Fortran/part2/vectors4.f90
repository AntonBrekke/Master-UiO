!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Remember in Fortran all arrays start at 1, as in science.
!
! Example of vector or indexed variables syntax.
! 
! 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 
!
program vectors
  use iso_fortran_env
  implicit none

  integer(int16) :: i
  integer(int16), parameter :: n=10
  integer(int16), dimension(n) :: a, b
  logical(int8), dimension(n)  :: c
  real(real32), dimension(n/2) :: x
  a=[-2, -3, -4, -1, 0, 1, 4, 2, 3, 5] ; b=0
  print '(10(i2x))', a

! Where is the zero ?
  print *,'Zero in a is at position : ', findloc(a, 0)  

! Making a mask vector.
  print *,'Mask vector c a>0 '
  c=(a>0)
  print *,c
  print *,'Any true values ? : ', any(c)
  print *,'All values true ? : ', all(c)
  print *,'All values true ? : ', all(c(6:10)) ! Subset of c elements 6 to 10.
  print *,'How many true ?   : ', count(c)
  
! The pack function
  print *,'Pack '
  b=pack(a,a>0) ! Only assign a(j) to b(j) where a(j)>0. 
  print '(10(i2x))', b

! The where masking
  print *,'Where masking'
  where (a<=1)
     b=a      ! Only assign b(j) to a(j) where a is less or equal to 1.
  end where
  print '(10(i2x))', b

! Taking the square root  
  print '(a)','Mask out negative for square root.'
  x=[-3.0, 9.0, -2.0, -6.0, 4.0]
  where (x>=0)
     x=sqrt(x)
  elsewhere
     x=sqrt(-x)
  end where
  print '(5(f6.3x))', x

end program vectors
