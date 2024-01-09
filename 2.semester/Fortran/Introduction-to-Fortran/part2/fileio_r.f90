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

program fileio
  use iso_fortran_env
  implicit none

  integer(int8) :: fileunit, filesize, iostatus
  logical :: present
  character(len=12) :: filename='datafile.dta'
  integer(int8) :: j, i

  inquire(file=filename, exist=present, size=filesize) ! Check if file exist.
  if (.not. present) then
     print *,"File not found"
     call exit(1)
  end if
  print *, filename,' has size: ',filesize,' bytes'
  open(newunit=fileunit, file=filename, action='read')


  ! We know there are 10 numbers.  
  do i=1,10
     read(fileunit,*) j 
     write(*,'(i3)') j
  end do

! File is still open and rear marker is at the end of the file.
! Position the file read position to the start, e.g. rewinding the tape.  
  rewind(fileunit)

! This time we don't know how many numbers there are in the file.
  do
     read(fileunit, fmt=*, iostat=iostatus) j
     if (is_iostat_end(iostatus)) exit  ! Intrisic function is_iostat_end check for EOF.
     write(*,'(i3)') j                  ! iostatus/=0 is simpler, but end og record or      
  end do                                ! errors will also trigger stop reading. 
  
  close(fileunit)  
  
end program fileio
