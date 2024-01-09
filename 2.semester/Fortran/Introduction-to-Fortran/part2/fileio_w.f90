!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Remember in Fortran all arrays start at 1, as in science.
!
! Example of file I/O
! 
! 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 

program fileiow
  use iso_fortran_env
  implicit none

  integer(int8) :: fileunit
  character(len=12) :: filename='datafile.dta'
  integer(int8) :: j

! Use newunit, return a number which is the fileunit.
! Older syntax unit needed a numerical constant.  
  open(newunit=fileunit, file=filename, action='write')

! We write 10 numbers.  
  write(fileunit,'(i3)') (j, j=1,10)

  write(*,*) 'Written 10 numbers to the file'
  close(fileunit)
       
end program fileiow
