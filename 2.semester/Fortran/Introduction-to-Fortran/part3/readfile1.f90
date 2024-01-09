!
!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! 
! Reading in a file with unknown lenght.  How do we get the number of lines ?
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 

program readfile
  use iso_fortran_env
  implicit none 

  character(len=30) :: filename='xy.txt'
  integer(int8) :: fn, iost
  integer(int32) :: j=0
  character(len=132) :: line
  logical(int8) :: present
  real(real64), allocatable, dimension(:) :: x,y
  
  inquire(file=filename, exist=present) ! Check if file exist.
  if (.not. present) then
     print *, 'File not found, exiting'
     call exit(1)
  end if
  open(newunit=fn, file=filename, status='old', action='read')
  do 
     read(fn, fmt="(a)", iostat=iost) line
     if (iost/=0) exit
     j=j+1
  end do
! We now know how many lines and number of x and ys.
  allocate(x(j),y(j))
  print *,'Allocated vectors size: ',size(x), size(y)

  
  rewind(fn) ! Spool the tape back to the beginning
  j=1
  do 
     read(fn,*,iostat=iost) x(j), y(j)
     if (iost/=0) then
          exit ! Exit do loop, EOF or error.
     end if
     j=j+1      
  end do
  close(fn)
    
  write(*,'(a,10f5.2)') 'x(1:10)', (x(j), j=1,10) ! Here comma is neeed after a, a10 is  
  write(*,'(a,10f5.2)') 'y(1:10)', (y(j), j=1,10) ! a string og length 10.

  
  deallocate(x,y)
end program readfile
  
  
