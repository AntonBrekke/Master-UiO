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
! in a pure subroutine. 

pure subroutine multwo(a,b,c)
  use iso_fortran_env
  implicit none
  integer(int8), intent(in) :: a,b ! Only in parameters, cannot be altered in subroutine.
  integer(int8), intent(out) :: c  
  
  c = a * b

end subroutine multwo



program mulfunctest
  use iso_fortran_env
  implicit none
  integer(int8) :: a,b,c

! We need to know how to call the function, an interface is needed.
! The interface provide type and size of input and output parameters.  
  interface
     pure subroutine multwo(a,b,c)
       use iso_fortran_env
       implicit none
       integer(int8), intent(in) :: a,b 
       integer(int8), intent(out) :: c  
     end subroutine multwo
  end interface

  
  a = 2
  b = 5
  print *, a, b
  
  call multwo(a,b,c)  

  print *, a,b,c

end program mulfunctest
