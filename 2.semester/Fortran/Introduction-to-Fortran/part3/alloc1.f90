
!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Remember in Fortran all arrays start at 1, as in science.
!
! Your fist memory allocation program in f90
! 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 

program alloc
  use iso_fortran_env
  implicit none
  
  real(real64), allocatable, dimension (:) :: x  ! Allocatable, dimension unknown ':'
  integer(int64) :: vectorsize

  write(*, '(a)',advance='no') 'How large vectors >'
  read(*,*) vectorsize

! We allocate a chunk of memory. Each element is a real of 32 or 64 bits, the f2008 std
! function storage_size(x) return the number of bits used by each element of x.
! To get the amount of memory used size(x)*storage_size(x)/8 .       

! GNU fortran support the C function sizeof(x) which report how many bytes the argument
! occupy in memory. But storage_size(x) is Fortran 2008 standard, reporting in bits.
! Be careful with GNU extentions, it might not be portable to other compilers. 

! We allocate space in memory for the vector.      
  allocate(x(vectorsize))

! The random subroutine support vector syntax.  
  call random_number(x)

  write(*,'(a,i3,a,i3,a,i3,a)') 'Size of vector x ', sizeof(x),&
       ' Size of datatypes in x ', storage_size(x), ' bits,',storage_size(x)/8,' Bytes'
  
  write(*,'(ai0ai0a)') 'Array of ',size(x),' take up ',size(x)*storage_size(x)/8,' bytes'
  write(*,'(10f5.2)') x(1:vectorsize:4) ! Every 4th element.
  write(*,'(a,2g12.4)') 'Sum of vector: ',sum(x), sum(x,x>0.5) ! Some vector functions take
                                                               ! mask argument
  write(*,'(a,l)') 'Is vector contiguos x       : ',is_contiguous(x)
  write(*,'(a,l)') 'Is vector contiguos x(1::2) : ',is_contiguous(x(1::2))
! We need to release memory.   
  deallocate(x)
end program alloc


