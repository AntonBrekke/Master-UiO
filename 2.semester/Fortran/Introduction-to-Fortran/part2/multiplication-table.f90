!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Remember in Fortran all arrays start at 1, as in science.
!
! Example of vector or indexed variables and implied do loops.
! 
! 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 
! Example of an implied do loop, (i, i=1,10)
!
program multab
  use iso_fortran_env
  implicit none

  integer(int8) :: i,j
  character(len=60) :: form


  write(*,*) 'Multiplication table:'
  form='(10(i3,x)/)'                  ! Ten (integers width 3 + one space) and a newline.
  do i=1,10
     write(*,form,advance='no') (i*j, j=1,10)  ! This is an implied do loop.
  end do

  
end program multab
