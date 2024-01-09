
!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Remember in Fortran all arrays start at 1, as in science.
!
! Stack and heap allocation
! 
! Compile gfortran -Dstatic -DN=200000000  alloc7.F90
! This uses 1.5 GiB memory.
!
! The shell command size show segment sizes:
! 
! gfortran -Dstatic -DN=200000000 -g  alloc6.F90
!    text    data     bss     dec     hex filename
!   2987     684 1600000032      1600003703      5f5e1e77        ./a.out
!
! gfortran  -DN=200000000 -g  alloc6.F90
!    text    data     bss     dec     hex filename
!    4056     724       4    4784    12b0 ./a.out
!
!
!
! Written by Ole W. Saastad, UiO
! October 2023.
! 

#ifndef N
#   define N    20000000
#endif

program alloc
  use iso_fortran_env
  implicit none

#ifdef static
  real(real64), dimension(N) :: x
#else
  real(real64), allocatable, dimension (:) :: x 
#endif  

! We allocate a chunk of memory, static on the stack or dynamic on the heap.
! Each element is a real of 32 or 64 bits, the f2008 std
! function storage_size(x) return the number of bits used by each element of x.
! To get the amount of memory used size(x)*storage_size(x)/8 .       

! GNU fortran support the C function sizeof(x) which report how many bytes the argument
! occupy in memory. But storage_size(x) is Fortran 2008 standard.        

#ifndef static
! We allocate space in memory for the vector.
  allocate(x(N))
#endif

! The random subroutine support vector syntax.  
  call random_number(x)

  write(*,'(ai0ai0a)') 'Array of ',size(x),' take up ',&
       size(x)/(1024*1024)*storage_size(x)/8,' Mbytes'
  
  write(*,'(3f5.2)') x(1), x(N/2), x(N) 

  write(*,'(a)', advance='no') 'Hit enter to continue'; read(*,*)
  
#ifndef static  
! We need to release memory.
  deallocate(x)
#endif

end program alloc
