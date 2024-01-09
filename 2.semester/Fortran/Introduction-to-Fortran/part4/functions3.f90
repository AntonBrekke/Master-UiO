!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Functions, do it right, more correctly
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

! We declare a function to multiply the argument by 2.
! We want no side effects and declate the function as pure.

pure function mulbytwo(a) result(b)
  use iso_fortran_env
  implicit none
  integer(int8), intent(in) :: a ! We need to add intention for pure functions.
  integer(int8) :: b

!  print *,a print is not allowed in pure functions.

  b = a * 2

!  a = 9 assigment of input variables are defintly not allowed. 
end function mulbytwo

! This is a simple function that works correctly, with no side effects.
! The syntax prevent ut from letting side effects pass the compilation step.


program mulfunctest
  use iso_fortran_env
  implicit none
  integer(int8) :: d,f
  integer(int8) :: mulbytwo  ! This is simple solution, not really correct. 
                             ! No check if parameters are corrent, we might try to
                             ! pass a real number or an integer of size different from 8.
  d = 2
  print *, d
  f = mulbytwo(d)  
  print *, f
  print *,'The "d" variable is :', d, ' ?'

end program mulfunctest