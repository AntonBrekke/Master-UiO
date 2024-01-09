!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Module with functions and subroutines.
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

module myroutines
  use iso_fortran_env
  implicit none
  private                  ! All are private.
! Except those who are declared public.
  public mulbytwo, multwo, fib   

contains
  pure function mulbytwo(a) result(b)
    integer(int8), intent(in) :: a ! We need to add intention for pure functions.
    integer(int8) :: b
    b = a * 2
  end function mulbytwo

  pure subroutine multwo(a,b,c)
    integer(int8), intent(in) :: a,b ! Only in parameters, cannot be altered in subroutine.
    integer(int8), intent(out) :: c  
    
    c = a * b
  end subroutine multwo

  recursive function fib(n) result(f)
    use iso_fortran_env
    implicit none
    integer(int32), intent(in) :: n 
    integer(int32) :: f
    
    if (n <= 2) then
       f = n
    else
       f = fib(n-1) + fib(n-2)
    end if    
  end function fib

  
end module myroutines

