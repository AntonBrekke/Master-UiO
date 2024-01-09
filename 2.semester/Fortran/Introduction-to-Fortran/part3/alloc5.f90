!
!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! 
!
! Memory allocation program in f90, polymorphism
!
! Use the CLASS(*) specifier to declare an unlimited polymorphic object.
! An unlimited polymorphic entity is not declared to have a type, and
! is not considered to have the same declared type as any other entity,
! including another unlimited polymorphic entity.
! 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 

program alloc
  use iso_fortran_env
  implicit none
  
  real(real64), allocatable, target, dimension(:)  :: x
  real(real32), allocatable, target, dimension(:)  :: y
  integer(int64), allocatable, target, dimension(:):: z
  class(*), pointer, dimension(:) :: pt   ! pt is a polymorph pointer, rank is fixed !
  integer(int32) :: status, u=1, vectorsize
  character :: ut

  write(*, '(a)',advance='no') 'How large vectors >'
  read(*,*) vectorsize

! We allocate a chunk of memory. Each element is a real of 32 or 64 bits, the f2009 std
! function storage_size(x) return the number of bits used by each element of x.
! To get the amount of memory used size(x)*storage_size(x)/8 .       

! We allocate space in memory for the vector.
  
  allocate(x(vectorsize), stat=status)
  if (status /= 0) then                     ! Two ways to handle error.
     print *,'Error in allocation'          ! If block, multiple statements.
     call exit(status)
  end if                                    ! Only one statement allowed.
  allocate(y(vectorsize/2), stat=status); if (status/=0) call exit(status) 
  allocate(z(vectorsize), stat=status); if (status/=0) call exit(status) 
  
  call random_number(x)    ! Vector syntax, all elements gets a random number.
  call random_number(y)
  z = int(x*100, int64)        ! Random return real, hence we just copy vector x to integer.
  
! Lets print out the array sizes in memory:  
  write(*,'(ai0ai0xa)') 'Array of ',size(x),' real64 take up ',&
       (size(x)*storage_size(x)/8)/u,'bytes'


  
! Let our polymorph pointer point to array x which is of type real64.
  pt => x

! We need to know the type before we can operate with the data.
  
  select type (pt)
  type is (real(kind=real32))
     write(*,'(6f5.2)') pt(1:3),pt(size(pt)-2:size(pt))
  type is (real(kind=real64))
     write(*,'(6f5.2)') pt(1:3),pt(size(pt)-2:size(pt))
  end select

  print *
! Let our polymorph pointer point to array x which is of type real32, a different type.
  pt => y
  select type (pt)
  type is (real(kind=real32))
     write(*,'(6f5.2)') pt(1:3),pt(size(pt)-2:size(pt))
  type is (real(kind=real64))
     write(*,'(6f5.2)') pt(1:3),pt(size(pt)-2:size(pt))
  end select
  
  print *
! Let our polymorph pointer point to array z which is of type integer, a different type.  
  pt => z
  select type(pt)
  type is (real(kind=real32))
     write(*,'(6f5.2)') pt(1:3),pt(size(pt)-2:size(pt))
  type is (real(kind=real64))
     write(*,'(6f5.2)') pt(1:3),pt(size(pt)-2:size(pt))
  type is (integer(kind=int64))
     write(*,'(6i5)')  pt(1:3),pt(size(pt)-2:size(pt))
  end select
  
! The polymorph pointer can point to a vector of any type.
  
  
  deallocate(x,y)
end program alloc


