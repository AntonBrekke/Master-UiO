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

program fileior
  use iso_fortran_env
  implicit none

  integer(int8) :: fileunit, iostatus
  integer(int64) :: filesize 
  logical :: present
  character(len=12) :: filename='datafile.dta'
  real(real64), dimension(1000,1000) :: a

  inquire(file=filename, exist=present, size=filesize) ! Check if file exist.
  if (.not. present) then
     print *,"File not found"
     call exit(1)
  end if
  print *, filename,' has size: ',filesize,' bytes'
  open(newunit=fileunit, file=filename, form="unformatted", action='read')
! newunit automatically provide a unit number, f2008 std.
  
  read(unit=fileunit, iostat=iostatus) a   ! How simple can it be ?
  if (iostatus/=0) then
     print *,'Error reading the binary file', iostatus
  end if
  print *,'Read the binary file'
  print '(10f5.2)', a(1:10,1)
  print '(f10.2)', sum(a)
  
  close(fileunit)  
  
end program fileior
