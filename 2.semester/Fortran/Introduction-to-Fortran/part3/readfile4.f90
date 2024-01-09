!
!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
!
! Memory allocation program in f90
!
! Chain of pointers, linked list.
!
! Reading in a file with unknown lenght into a linked list of line objects.
! The issue with this approach is 2x memory is needed of which
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 

program readfile
  use iso_fortran_env
  implicit none 

  type line
     real(real32) :: x, y
     type(line), pointer :: next_line
  end type line
  
  character(len=30) :: filename='xy.txt'
  integer(int16) :: fn, iost
  integer(int32) :: fnsize, fnlines=0, j=1
  logical(int8) :: present
  real(real32) :: xi ,yi
  real(real32), allocatable, dimension(:) :: x,y
  type(line), pointer :: start, traverse

  allocate(start)
  traverse => start
  nullify(traverse%next_line)
  
  inquire(file=filename, exist=present, size=fnsize) 
  if (.not. present) then
     print *, 'File not found, exiting'
     call exit(1)
  end if
  write(*,'(a,a,i5,a)') 'Size: ',trim(filename), fnsize, ' bytes.'
  open(newunit=fn, file=filename, status='old', action='read')

  do 
     read(fn,*,iostat=iost) traverse%x, traverse%y
     if (is_iostat_end(iost)) exit   ! Intrisic function is_iostat_end check for EOF.
     allocate(traverse%next_line)    ! allocate next object       
     traverse => traverse%next_line  ! move to next in the list.
     fnlines = fnlines + 1           
  end do
  nullify(traverse%next_line) ! Nullify next_line as this is invalid.
  print *,'EOF'

! Print out the data to check that it was read ok.
! Traverse the linked list.
  traverse => start     ! Start at beginning
  do
     print '(2f6.2)', traverse%x, traverse%y
     traverse => traverse%next_line
     if (.not. associated(traverse%next_line)) exit
  end do

! We know how many lines read from the file. We can allocate two
! arrays x and y with the correct size. 
 
  allocate(x(fnlines))
  allocate(y(fnlines))

  traverse => start
  do
     x(j) = traverse%x
     y(j) = traverse%y
     traverse => traverse%next_line
     if (.not. associated(traverse%next_line)) exit
     j = j + 1
  end do
     
  write(*,'(a,10f5.2)') 'x(1:10)', (x(j), j=1,10) ! Here comma is neeed after a, a10 is  
  write(*,'(a,10f5.2)') 'y(1:10)', (y(j), j=1,10) ! a string og length 10.

  traverse => start
  do j=1, fnlines
     start => traverse
     traverse => traverse%next_line
     deallocate(start)
!     print *, associated(start), associated(traverse)
  end do
  deallocate(traverse)
!  print *, associated(start), associated(traverse)

!  We now have only vectors x and y using memory. 

  deallocate(x,y)
end program readfile
  
  
