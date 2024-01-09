!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Arguments with unknown rank.
!
! Ole W. Saastad, UiO.
! November 2023
!
!
! module load  GCC/12.2.0 to get 2008 standard
!
! The 12.2.0 GNU Fortran implements the Fortran 77, 90 and 95 standards completely, 
! most of the Fortran 2003 and 2008 standards, and some features from the 2018 standard. 
!
! It’s nice that in Fortran 2018, a function can have an assumed-rank array argument that
! can be a scalar or an array of any rank.



function addv(a) result(r)
  use iso_fortran_env
  implicit none
  integer(int32), intent(in), dimension(..) :: a
  integer(int32) :: r

! I/O is allowed in impure functions.   
  print *, 'Function addv'
  print *, 'Rank of argument ', rank(a)
  print *, 'Shape of argument ', shape(a)  

! We need to handle each rank size individually.  
  select rank(a)
    rank(0); r = a
    rank(1); r = sum(a)
    rank(2); r = sum(a)
    rank(3); r = sum(a)
    rank(4); r = sum(a)
    rank default ; r = -huge(a)
  end select
end function addv


subroutine flatten(x, y)  ! Flatten any dimensional array into a 1-d vector.
  use iso_fortran_env     ! only up to rank 3 in this implementation.
  implicit none
  integer(int32), intent(in), dimension(..) :: x   ! Syntax for unknown rank.
  integer(int32), intent(inout), dimension(size(x)) :: y

! I/O is allowed in impure functions.     
  print *, 'Subroutine flatten'
  print *, 'size x,y ',size(x), size(y)
  print *, 'rank of x ',rank(x)
  print *, 'shape of x ',shape(x)

  select rank(x)
    rank(1) ; y = reshape(x, shape=[size(x)])
    rank(2) ; y = reshape(x, shape=[size(x)])
    rank(3) ; y = reshape(x, shape=[size(x)])
    rank default ; y=0
  end select
  print *, 'size of y ', size(y)
end subroutine flatten





program assumedranktest
  use iso_fortran_env
  implicit none
  integer(int32), dimension(2,2,2) :: a
  integer(int32), dimension(:), allocatable :: b
  integer(int32) :: s

  
! We need an interface here to inform how to call the routines.
  interface
     function addv(a) result(r)
       use iso_fortran_env
       implicit none
       integer(int32), intent(in), dimension(..) :: a
       integer(int32) :: r
     end function addv
     
     subroutine flatten(x, y)
       use iso_fortran_env
       implicit none
       integer(int32), intent(in), dimension(..) :: x
       integer(int32), intent(inout),  dimension(size(x)) :: y
     end subroutine flatten     
  end interface

  
! Main block start here
  a = 1
  print '(a)', 'Function addv '

  print '(ai2)', 'rank of a  ',rank(a)
  print '(a,4i2)', 'shape of a ', shape(a)
  s = addv(a)
  print '(ai3)','Sum vector ',s
  print *

  a = 2
  allocate(b(size(a)))
  print '(a)', 'Subroutine flatten '
  print '(ai4)', 'size of a ' ,size(a)
  print '(ai4)', 'size of b ' ,size(b)
  print '(ai1,a,4i2)', 'rank of a: ',rank(a), ', shape of a ', shape(a)
  
  call flatten(a,b) ! Flatten any rank array into a 1-d vector.
  
  print '(ai2,a,4i2)', 'rank of b  ',rank(b), ', shape of b ', shape(b)
  print '(a,i2)',  'size of b ' ,size(b)
  print '(a,4i2)', 'b(1:4): ', b(1:4)

  deallocate(b)
end program assumedranktest
