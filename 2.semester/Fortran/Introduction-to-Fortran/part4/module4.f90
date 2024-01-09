!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Use routines from a module
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
! How does a function or a program know the way of using a function? A function has a name,
! some arguments with certain types, and the type of the function value? Without these 
! information, a function might not be used correctly. To overcome this problem, an interface
! block is introduced. More precisely, any external function to be used should be listed in
! an interface block along with the declaration of its arguments and their types and the type
! of the function value. 




program moduletest
  use myroutines
  use iso_fortran_env
  implicit none
  
  integer(int8) :: d,f,g

  d=6
  print *,d
  f = mulbytwo(d)
  print *, f

  call multwo(d,f,g)
  print *,g
  
end program moduletest
