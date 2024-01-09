!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Remember in Fortran all arrays start at 1, as in science.
!
! Example of vector or indexed variables and implied do loops.
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
  integer(int16), parameter :: n=3
  real(real32), dimension(n) :: a
  real(real64), dimension(n,n) :: b
  real(real64), dimension(n,n,n) :: c
  
  print *, 'size a: ', size(a)
  print *, 'size b: ', size(b)
  print *, 'size c: ', size(c)

! Vector syntax:
  
  a=2.0_real32
  b=3.0_real64
  c=0.75_real64

  print *,'Columns and rows, fortran is column major, first index is column'
  
  print *,'One vector :'       
  print '(3f5.2)', a(:)         ! The colon is like a wildcard, meaning all entries.
  print '(3f5.2)', b(:,1)       ! Only line 1
  print '(3f5.2)', c(:,1,1)


  b(:,1) = 0.2_real64           ! Assign only line 1.
  b(:,2) = 0.6_real64           ! Assign only line 2.
  print *
  print *,'b columns 1-3 line 1:'
  print '(3f5.2)', b(1,1), b(2,1), b(3,1)
  print *,'b column 1 line 1-3:'
  print '(f5.2)', b(1,1), b(1,2), b(1,3)
  print *
  print '(a/,3(f5.2x))','b :',b
  print *

  print *,'More formatting:'
  a=sqrt(a)
  print '(a/,3f5.2)', 'a :', a
  b=log(b)
  print '(a/,3(f5.2x))','b :',b
  c=sin(c)
  print '(a/,3(f5.2x))', 'c :', c
  print '(a/,3(3(f5.2x)2x))', 'c :', c(:,:,1), c(:,:,2), c(:,:,3)
end program vectors
