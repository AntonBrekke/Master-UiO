
!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Dynamic allocation and pointers.
!
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 

program alloc
  use iso_fortran_env
  implicit none
  
  real(real64), allocatable, target, dimension (:) :: x,y  ! Allocatable, dimension unknown ':'
  real(real64), pointer, dimension(:) :: pt
  integer(int32) :: vectorsize

  write(*, '(a)',advance='no') 'How large vectors >'
  read(*,*) vectorsize

! We allocate a chunk of memory. Each element is a real of 32 or 64 bits, the F2008 std
! function storage_size(x) return the number of bits used by each element of x.
! To get the amount of memory used size(x)*storage_size(x)/8 .       

! We allocate space in memory for the vector.
  
  allocate(x(vectorsize), y(vectorsize/2))
  
  call random_number(x)
  call random_number(y)

  write(*,'(ai0ai0a)') 'Array of ',size(x),' take up ',size(x)*storage_size(x)/8,' bytes'
  
! Let pointer point to array x:
  pt => x
  write(*,'(ai0ai0a)') 'Array of ',size(pt),' take up ',size(pt)*storage_size(pt)/8,' bytes'
  write(*,'(2f5.2)') pt(1),pt(size(pt))

! Let pointer point to array y which have half the size of x  
  pt => y
  write(*,'(ai0ai0a)') 'Array of ',size(pt),' take up ',size(pt)*storage_size(pt)/8,' bytes'
  write(*,'(2f5.2)') pt(1),pt(size(pt)) 


! We need to release memory.   
  deallocate(x,y)
end program alloc


