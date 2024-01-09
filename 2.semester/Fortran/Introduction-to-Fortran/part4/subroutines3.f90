!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Subroutines, do it right, more correctly
!
! Ole W. Saastad, UiO.
! September 2023
!
!
!  module load  GCC/12.2.0 to get 2008 standard
!
! The 12.2.0 GNU Fortran implements the Fortran 77, 90 and 95 standards completely, 
! most of the Fortran 2003 and 2008 standards, and some features from the 2018 standard. 
!
!

! We declare a subroutine to multiply two arguments and return the result as
! an out parameter. 
! We want no side effects and declate the subroutine as pure. No I/O is allowed
! in a pure subroutine, so the two subroutines here are not pure.

subroutine multwo1(a,b,c)
  use iso_fortran_env
  implicit none
  integer(int8), dimension(:), intent(in) :: a,b ! Only in parameters
  integer(int8), dimension(:), intent(out) :: c  

  print *, 'In Subroutine 1, are vectors contiguos? ',is_contiguous(a),is_contiguous(b),&
       is_contiguous(c)
  c = a * b

end subroutine multwo1


subroutine multwo2(a,b,c)
  use iso_fortran_env
  implicit none
  integer(int8), contiguous, dimension(:), intent(in) :: a,b ! Only in parameters
  integer(int8), contiguous, dimension(:), intent(out) :: c  

  print *, 'In Subroutine 2, are vectors contiguos? ',is_contiguous(a),is_contiguous(b),&
       is_contiguous(c)
  
  c = a * b

end subroutine multwo2




program mulfunctest
  use iso_fortran_env
  implicit none
  integer(int8), dimension(:), allocatable :: a,b,c

! We need to know how to call the function, an interface is needed.
! The interface provide type and size of input and output parameters.  
  interface
     subroutine multwo1(a,b,c)
       use iso_fortran_env
       implicit none
       integer(int8), dimension(:), intent(in) :: a,b 
       integer(int8), dimension(:), intent(out) :: c  
     end subroutine multwo1
     subroutine multwo2(a,b,c)
       use iso_fortran_env
       implicit none
       integer(int8), contiguous, dimension(:), intent(in) :: a,b 
       integer(int8), contiguous, dimension(:), intent(out) :: c  
     end subroutine multwo2
     
  end interface

  allocate(a(6),b(6),c(6))
  a = 2
  b = 5
  c = 0
  print *, a,';', b
  print *, 'Are vectors a and b contiguos? ',is_contiguous(a),is_contiguous(b)
  print *, 'Calling subroutines with every second element'
  
  call multwo1(a(1::2),b(1::2),c(1::2))  
  print *, a,';',b,';',c

  call multwo2(a(1::2),b(1::2),c(1::2))  
  print *, a,';',b,';',c
  
  
end program mulfunctest
