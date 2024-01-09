!
!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! 
!
! Memory allocation program in f90
!
! Reading in a file with unknown lenght, converting to unformatted.
! This method only require memory to store the x and y vectors in 64bit format.
! We need to read the ascii file and write it as unformatted, but we only need
! memory to fit the data once.
!
! 'newunit' is a new way of getting a new unit number which is free, no reason to
! know it's value. 
!
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 

program readfile
  use iso_fortran_env
  implicit none 

  character(len=30) :: filename='xy.txt'
  integer(int16) :: fn, fnu, iost
  integer(int32) :: fnsize, j=0, elements=0
  logical(int8) :: present
  real(real64) ::  xi ,yi
  real(real64), allocatable, dimension(:) :: x,y

! Options to use command line arguments.
  
  if (command_argument_count() < 1) then
     print *,'No input file, use default: ', filename
  else
     call get_command_argument(1, filename)
  end if


  inquire(file=filename, exist=present, size=fnsize) 
  if (.not. present) then
     print *, 'File not found, exiting'
     call exit(2) ! Exit code 2 is misuse.
  end if
  write(*,'(a,a,i5,a)') 'Size: ',trim(filename), fnsize, ' bytes.'

! Open the ascii data file  
  open(newunit=fn, file=filename, status='old', action='read')
! Open the binary raw file.
  open(newunit=fnu, file='raw.dta', status='unknown', form='unformatted')

  do
     read(fn,*,iostat=iost) xi, yi
     if (is_iostat_end(iost)) exit  ! Intrisic function is_iostat_end check for EOF.
     write(fnu) xi, yi        ! Write data in binary form, raw unformatted.
     elements = elements + 1  
  end do
  close(fn)
! We have now  read the ascii and written the data in binary format.
  
  allocate(x(elements), y(elements))  ! We know how many elements and allocate vectors.
  print *, size(x), size(y)           ! Check correct size ? 

  rewind(fnu)   ! Old statement, rewind tape. 
  do j=1,size(x)     
     read(fnu) x(j), y(j) 
     write(*,'(i4,x,2f6.2)') j, x(j), y(j)
  end do
  close(fnu)

  deallocate(x,y)
end program readfile

