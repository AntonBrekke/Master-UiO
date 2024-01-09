!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!
! Example of implied do loops.
! 
! 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 
! Example of an implied do loop, implied to loops are a very
! powerful contruct. 

program implied
  use iso_fortran_env
  implicit none

  integer(int8) :: i,j

  print *,'One implied loop'
  print *,(i, i=1,10)

  print *,'Two implied loops'
  write(*,'(10i3)') ((i+j, i=1,10), j=1,10)

  print *,'Three uses of loop counter'
  print '(i3,x,i3,x,i4)', (i, i+i, i*i, i=-5,4)
end program implied
