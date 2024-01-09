!
!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Remember in Fortran all arrays start at 1, as in science.
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
  
  if (vectorsize > 1024 .and. vectorsize < 1024*1024) then
     u=1024 ; ut='k'
  elseif (vectorsize > 1024*1024) then
     u=1024*1024 ; ut='M'
  end if
  
! More on allocation. 
!
! Allocate can use source and mold. Both use yield a new array of same dimension and size as
! the source/mold one. Source will copy the data from the source vector while mold only copy   
! the dimensions. You can also provide a constant as source, but then the size need to be supplied.
!
! https://fortran-lang.discourse.group/t/using-source-and-mold-for-multiple-arrays/945
!
! https://www.ibm.com/docs/en/xffbg/121.141?topic=attributes-allocate
!

  
  write(*,'(/a)')  'allocate source and mold'

  pt => x
  allocate(z, source=pt)
  write(*,*) 'Source copy data, vector x'
  write(*,'(ai0ai0xaa)') 'Array of ',size(z),' take up ',&
       (size(z)*storage_size(z)/8)/u,ut,'bytes'
  write(*,'(2f5.2)') z(1),z(size(z)) 
  deallocate(z)

  pt => x
  allocate(z, mold=pt) ! We can also use the vector x as mold. 
  write(*,*) 'Mold does not copy data, vector x'
  write(*,'(ai0ai0xaa)') 'Array of ',size(z),' take up ',&
       (size(z)*storage_size(z)/8)/u,ut,'bytes'
  write(*,'(2f5.2)') z(1),z(size(pt)) 
  deallocate(z)
  
  pt => y
  allocate(z, mold=pt, stat=status); if (status/=0) error stop 'Allocation failed'
  write(*,*) 'Mold does not copy data, vector y'             ! program return /= 0 
  write(*,'(ai0ai0xaa)') 'Array of ',size(z),' take up ',&
       (size(z)*storage_size(z)/8)/u,ut,'bytes'
  write(*,'(2f5.2)') z(1),z(size(z)) 
  deallocate(z)

! x(1::4) is x(1) to x(n) every fourth element vector syntax is powerful. 
  allocate(z, mold=x(1::4), stat=status); if (status/=0) error stop 'Allocation failed'
  write(*,*) 'Mold subset of vector x'                   ! program return /= 0 
  write(*,'(ai0ai0xaa)') 'Array of ',size(z),' take up ',&
       (size(z)*storage_size(z)/8)/u,ut,'bytes'
  write(*,'(2f5.2)') z(1),z(size(z)) 
  deallocate(z)
 
  
  allocate(z(vectorsize), source=3.14_real64)
  write(*,*) 'Source copy data, constants'
  write(*,'(ai0ai0xaa)') 'Array of ',size(z),' take up ',&
       (size(z)*storage_size(z)/8)/u,ut,'bytes'
  write(*,'(2f5.2)') z(1),z(size(z)) 
  deallocate(z)


! Combination of allocate and pack with mask.
! Read more on pack and related functions:
! https://gcc.gnu.org/onlinedocs/gfortran/PACK.html

  allocate(z(size(pack(x,x<0.5))), source=pack(x,x<0.5))
  write(*,*) 'Source copy data, constants'
  write(*,'(ai0ai0xaa)') 'Array of ',size(z),' take up ',&
       (size(z)*storage_size(z)/8)/u,ut,'bytes'
  write(*,'(2f5.2)') z(1),z(size(z)) 
  deallocate(z)
  
  
! We should have used a function called printout, functions are next session.
  
! We need to release memory.  
  deallocate(x,y)
end program alloc


