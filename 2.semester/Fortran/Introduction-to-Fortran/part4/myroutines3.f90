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
  private           ! All are private.
  public mulbytwo   ! Except those who are declared public.

contains
  pure function mulbytwo(a) result(b)
    integer(int8), intent(in) :: a ! We need to add intention for pure functions.
    integer(int8) :: b
    b = a * 2
  end function mulbytwo
  
end module myroutines

