!
!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!
! Memory allocation program in f90, source and mold. 
! 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 

program alloc
  use iso_fortran_env
  implicit none
  
  real(real64), allocatable, target, dimension (:) :: x,y,z  ! Allocatable, dimension unknown ':'
  real(real64), pointer, dimension(:) :: pt       ! pt can only point to same type and dimension.
  integer(int32) :: status, u=1, vectorsize
  character :: ut

  write(*, '(a)',advance='no') 'How large vectors >'
  read(*,*) vectorsize

! We allocate a chunk of memory. Each element is a real of 32 or 64 bits, the f2009 std
! function storage_size(x) return the number of bits used by each element of x.
! To get the amount of memory used size(x)*storage_size(x)/8 .       

! We allocate space in memory for the vector.

! Two ways to handle error.
! If block, multiple statements.  
  allocate(x(vectorsize), stat=status)
  if (status /= 0) then                    
     print *,'Error in allocation'          
     call exit(status)
  end if                                    
! Only one statement after test without then.
  allocate(y(vectorsize/2), stat=status);
  if (status/=0) call exit(status) 

! Most intrinsic functions are vector aware.  
  call random_number(x)    ! Vector syntax, all elements gets a random number.
  call random_number(y)

! Print out i kiB or MiB if large, easier to read by humans.  
  if (vectorsize > 1024 .and. vectorsize < 1024*1024) then
     u=1024 ; ut='k'
  elseif (vectorsize > 1024*1024) then
     u=1024*1024 ; ut='M'
  end if

! Lets print out the array sizes in memory:  
  write(*,'(ai0ai0xaa)') 'Array of ',size(x),' take up ',&
       (size(x)*storage_size(x)/8)/u,ut,'bytes'

  
! Let pointer point to array x, here we can have different sizes of arrays. 
  pt => x
  write(*,'(ai0ai0xaa)') 'Array of ',size(pt),' take up ',&
       (size(pt)*storage_size(pt)/8)/u,ut,'bytes'
  write(*,'(2f5.2)') pt(1),pt(size(pt))

  
! Let pointer point to array y which have half the size of x  
  pt => y
  write(*,'(ai0ai0xaa)') 'Array of ',size(pt),' take up ',&
       (size(pt)*storage_size(pt)/8)/u,ut,'bytes'
  write(*,'(2f5.2)') pt(1),pt(size(pt)) 


  
! We should have used a function called printout, functions are next session.
  
! We need to release memory.  
  deallocate(x,y)
end program alloc


