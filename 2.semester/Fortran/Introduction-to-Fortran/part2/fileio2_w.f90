!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Remember in Fortran all arrays start at 1, as in science.
!
! Example of file I/O.
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
  integer(int16) :: i,j 
  character(len=12) :: filename='datafile.dta'
  real(real64), dimension(1000,1000) :: a

  call random_number(a)
  
  print '(10f5.2)', a(1:10,1)
  print '(f10.2)', sum(a)
  
  open(newunit=fileunit, file=filename, form="unformatted", action='write')
! newunit will assign a unit number for us automatically.
  
! We write the matrix a as binaray, just like it's stored in memory.
  write(fileunit) a                       
  write(*,*) 'Written binary data to file'
  close(fileunit)
  
end program fileiow
