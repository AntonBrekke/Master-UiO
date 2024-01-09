!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!
!
! Vector syntax - Pointers and arrays
!
! Ole W. Saastad, February 2023
!
!

program allocatable
  use iso_fortran_env
  implicit none
  integer(int8) :: j
  integer(int16), target,  allocatable, dimension(:) :: a
  integer(int16), target,  dimension(7)              :: b
  integer(int16), pointer, dimension(:)              :: p, q  
! Pointers are strongly typed, type and dimension must be known. 
! (:) for 1-d, (:,:) for 2-d etc.
 
! There is no void pointer casting of type is not allowed.

  allocate(a(5)) 
  
  print *, "Allocatable vectors, pointers and targets"
  print *
  print *, "Pointers associated or just null ?"
  print *,'Associated p, q ', associated(p), associated(q)
! Check if p is already associated with something.
  if (.not. associated(p)) then
     p => a ! pointer p is set to point to a. 
  end if
  print *,'Associated p, q ', associated(p), associated(q)
  write(*,'(a)', advance='no') 'Hit enter to continue'; read(*,*)
  
  print *, (j, j=1,5)  ! (/   /) == [  ]
  p = [ (j, j=1,5) ] ! Fill it with 1,2,3,4,5 ; implied do loop, fill the elements
                     ! of the vector defined using [ d,d,d,d,d ]
  print *, p         
! Expressions are allowed with implied do loops.
  p = [ (j*j, j=1,5) ]
  print *, p
  print *,'p point to a, size ', size(p)," : ", p 
  write(*,'(a)', advance='no') 'Hit enter to continue'; read(*,*)
  
  q => p ! Set pointer q to pointer p's target, they now point to the same target.
  nullify(p) ! p now point to null.
  print *,'Associated p, q ', associated(p), associated(q)
  q = 2 ! This is vector syntax, all elements in vector a is set to 2.
  print *,'p point to a, size ', size(q)," : ", q 
  nullify(q)
  write(*,'(a)', advance='no') 'Hit enter to continue'; read(*,*)
  
  p => b
  print *,'Associated p, q ', associated(p), associated(q)
  p = 3 
  print *,'p point to b, size ', size(p)," : ", p 
  print *,'shape of b ',shape(b)
  
  deallocate(a)
  write(*,'(a)', advance='no') 'Hit enter to continue'; read(*,*)
  p => a
  allocate(p(8)) ! We reallocate a to size 8 using the pointer.
  p = 4
  print *,'Associated p, q ', associated(p), associated(q)
  print *,'p point to a, size ', size(p)," : ", p 
    
end program allocatable



