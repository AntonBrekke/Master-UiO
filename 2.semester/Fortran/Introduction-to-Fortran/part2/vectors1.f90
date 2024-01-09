!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Remember in Fortran all arrays start at 1, as in science.
!
! Example of vector or indexed variables usage of : as range 1:3 is 1 through 3.
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

  integer(int8) :: i
  integer(int8), parameter :: n=3
  integer(int8), dimension(n) :: a
  integer(int8), dimension(n,n) :: b

  print *, 'Size of vector, number of elements'
  print *, 'size a         : ', size(a)
  print *, 'size b         : ', size(b)
  print *, 'size b(1:2,2:3): ', size(b(1:2,2:3))
! storage_size(x) return number of bits in each element, int8 equal 8 bits, one byte.   
  print *, 'Storage requirement of vector a, Bytes :',storage_size(a)*size(a)/8
  print *
  
! This is vector syntax in fortran, assigning all elements in one statement.   
  a=2
  b=3
  print *,'Extract and print some data'
  print *, a, '<- vector a'
  print *, b, '<- vector b'
  print *, a(1:2),  '<- Extracting only the two first numbers of a.'
  print *, a(1:2:2),'<- Extracting every second element. 1:2 is a 2 element vector, 1+2=3 => no number.'
  print *, b(1:2,2:3), '<- a 2x2 matrix extracted from b.'

  print *
! Can we print a matrix ?  Format statement comes to our help.
! We know n=3, hence the 3, after each formatted line a linefeed is emitted.

  print *,'Print a matrix'
  print '(3i2)', b  ! Voila, we printed a matix.
  print *
  print '(2i2)', b(1:2,2:3)
 
  
end program vectors
