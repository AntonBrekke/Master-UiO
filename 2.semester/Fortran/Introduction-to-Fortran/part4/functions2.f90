!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Functions, first approach. 
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

function mulbytwo(a) result(b)
  use iso_fortran_env
  implicit none
  integer(int8) :: a ,b

  print *,a

  b = a * 2

  a = 9
end function mulbytwo

! This is a simple function that works correctly, but some issues.
! It has side effects, it can print which might not be safe.
! As Fortran is by reference the a is a reference to memory
! and there is noting to prevent us from updating a within the
! function.

! Setting a which is the IN parameter to 9 works and represent a very
! unpleasant surprise for the non suspecting user. An unwated side
! effect. 


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
  print *,'How did this "d" variable suddenly become :', d, ' ?'

end program mulfunctest
