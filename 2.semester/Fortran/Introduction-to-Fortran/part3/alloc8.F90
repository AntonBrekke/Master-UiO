
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
! Compile gfortran -Dstatic -DN=200000000  alloc8.F90
! This uses 1.5 GiB memory.
!
! Written by Ole W. Saastad, UiO
! October 2023.
! 

#ifndef N
#   define N    20000000  
#endif
! This default is about 152 MiB.



program alloc
  use iso_fortran_env
  implicit none

#ifdef static
  real(real64), dimension(N) :: x
#else  
  real(real64), allocatable, dimension (:) :: x 
#endif
  real(real32) :: start_time, end_time

! We run two fillings of the vector using random_number
   
#ifdef static
! Static allocation block:  
  call cpu_time(start_time)
  write(*,'(ai0ai0a)') 'Array of ',size(x),' take up ',&
       size(x)/(1024*1024)*storage_size(x)/8,' Mbytes'
  call random_number(x)
  write(*,'(3f5.2)') x(1), x(N/2), x(N) 
  call random_number(x)
  write(*,'(3f5.2)') x(1), x(N/2), x(N)
  call cpu_time(end_time)
  write(*,'(a,f6.3,a)') 'CPU time static allocation : ', end_time-start_time, ' seconds'
#else
! Dynamic allocation block
  call cpu_time(start_time)
  allocate(x(N))  
  write(*,'(ai0ai0a)') 'Array of ',size(x),' take up ',&
       size(x)/(1024*1024)*storage_size(x)/8,' Mbytes'
  call random_number(x)
  write(*,'(3f5.2)') x(1), x(N/2), x(N) 
  deallocate(x)
  allocate(x(N))
  call random_number(x)
  write(*,'(3f5.2)') x(1), x(N/2), x(N)
  deallocate(x)
  call cpu_time(end_time)
  write(*,'(a,f6.3,a)') 'CPU time dynamic allocation : ', end_time-start_time, ' seconds'
#endif  

end program alloc
