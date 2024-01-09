!
!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! 
!
! Reading in a file with unknown lenght.  Using shell commands to get length. 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 

program readfile
  use iso_fortran_env
  implicit none 

  character(len=30) :: filename='xy.txt'
  integer(int16) :: fn, iost
  integer(int32) :: fnsize, fnlines, j=0
  character(len=132) :: cmd, line
  logical(int8) :: present
  real(real64) :: xi ,yi
  real(real64), allocatable, dimension(:) :: x,y
  
  inquire(file=filename, exist=present, size=fnsize) 
  if (.not. present) then
     print *, 'File not found, exiting'
     call exit(1)
  end if
  write(*,'(a,a,i5,a)') 'Size: ',trim(filename), fnsize, ' bytes.'

! There is no option with inquire to report how many lines there are in a file,
! we need to count the number of linefeeds in the file.  
  
  cmd='wc -l <'//filename//'>fntmp'  ! Execute this shell command, this also
  call execute_command_line(cmd)     ! reads the entire file, e.g. reading it twice. 
  open(newunit=fn, file='fntmp')
  read(fn,*) fnlines
  close(fn)
  call execute_command_line('rm fntmp') 
  write(*,'(a,i0,a)') 'There are ',fnlines,' in the input file'
 
  open(newunit=fn, file=filename, status='old', action='read')

  allocate(x(fnlines),y(fnlines))
  
  do j=1, fnlines
     read(fn,*,iostat=iost) x(j), y(j)
     if (iost/=0) then
        exit ! Exit do loop, EOF or error, we keep what we have and continue.
     end if
  end do
  
  print *,'Allocated vectors size: ',size(x), size(y)
  write(*,'(a,10f5.2)') 'x(1:10)', (x(j), j=1,10) ! Here comma is neeed after a, a10 is  
  write(*,'(a,10f5.2)') 'y(1:10)', (y(j), j=1,10) ! a string og length 10.

  deallocate(x,y)
end program readfile
  
  
