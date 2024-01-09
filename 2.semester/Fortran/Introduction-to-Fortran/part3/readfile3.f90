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
! Reading in a file with unknown lenght.  
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 

program readfile
  use iso_fortran_env
  implicit none 

  character(len=30) :: filename='xy.txt'
  integer(int16) :: fn, iost
  integer(int32) :: fnsize, j=0
  character(len=132) :: line
  logical(int8) :: present
  real(real64) :: xi ,yi
  real(real64), allocatable, dimension(:) :: x,y
  
  inquire(file=filename, exist=present, size=fnsize) 
  if (.not. present) then
     print *, 'File not found, exiting'
     call exit(1)
  end if
  write(*,'(a,a,i5,a)') 'Size: ',trim(filename), fnsize, ' bytes.'
  open(newunit=fn, file=filename, status='old', action='read')

  allocate(x(0),y(0))   ! Allocate zero elements.
  
  do 
     read(fn,*,iostat=iost) xi, yi
     if (iost/=0) then
        exit ! Exit do loop, EOF or error, we keep what we have and continue.
     end if
     
     x = [ x, xi ]  ! We append a new element to the vector.
     y = [ y, yi ]  ! A bit slow, reallocate and deallocate. 
                    ! An elegant solution, but check performance.
  end do
  
  print *,'Allocated vectors size: ',size(x), size(y)
  write(*,'(a,10f5.2)') 'x(1:10)', x(1:10) 
  write(*,'(a,10f5.2)') 'y(1:10)', y(1:10)

  deallocate(x,y)
end program readfile
  
  
