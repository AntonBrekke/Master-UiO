!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Overflow
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


program overflow
  use iso_fortran_env
  implicit none
  integer(int8) :: ioresult=1

  real(real64) :: x
  integer(int8) :: a

  write(*,'(a)') 'Integer overflow test'
  do while (ioresult /= 0)
     write(*,'(a)', advance='no') 'Give an number >'
     read(*,*, iostat=ioresult) x
  end do
  a=int(x)
  write(*,'(I8A)') a, '  decimal'
  write(*,'(B9.8A)') a, ' binary'
  write(*,'(AI2)') 'popcount ', popcnt(a)
end program overflow
