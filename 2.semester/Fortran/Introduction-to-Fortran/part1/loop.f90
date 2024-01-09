!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Control flow, loops
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

program loop
  use iso_fortran_env

  implicit none
  integer(int8)   :: i, n=5 

  print *, 'simplest do loop, infinite loop'
  i=1
  do
     print *,i
     i=i+1
     if (i>4) exit
  end do

  
  print *,'Simple loop'
! Simple loops
  do i=1,n
     print *,'Count ',i
  end do

  do while (i<10)
     write(*,'(i3)') i
     i=i+1
  end do

  
  print *,'Fixed loop'
! Loop length is fixed at start
  do i=1,n
     write(*,'(2(i3))') i, n
     n=0
  end do

  
end program loop
