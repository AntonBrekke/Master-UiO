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
!


pure function addv(a) result(r)
  use iso_fortran_env
  implicit none
  integer(int32), intent(in), dimension(..) :: a
  integer(int32) :: r

! One need to specify which rank size to operate on.  
  select rank(a)
    rank(0); r = a
    rank(1); r = sum(a)
    rank(2); r = sum(a)
    rank(3); r = sum(a)
    rank(4); r = sum(a)
    rank default ; r = -huge(a)  ! To do it right we should have extended to 15.
  end select
end function addv


pure subroutine flatten(x, y)
  use iso_fortran_env
  implicit none
  integer(int32), intent(in), dimension(..) :: x
  integer(int32), intent(inout), dimension(size(x)) :: y

! One need to specify which rank size to operate on.    
  select rank(x)
    rank(1) ; y = reshape(x, shape=[size(x)])
    rank(2) ; y = reshape(x, shape=[size(x)])
    rank(3) ; y = reshape(x, shape=[size(x)])
    rank(4) ; y = reshape(x, shape=[size(x)])
    rank default ; y=0 ! To do it right we should have extended to 15.
  end select
 
end subroutine flatten

! ROUTINES END HERE



! PROGRAM START HERE: 

program assumedranktest
  use iso_fortran_env
  implicit none
  integer(int32) :: j
  integer(int32), dimension(2,3,4) :: a
  integer(int32), dimension(:), allocatable :: b
  integer(int32) :: s
  integer(int8), allocatable, dimension(:) :: l
  
! We need an interface here to inform how to call routines.
  interface
     pure function addv(a) result(r)
       use iso_fortran_env
       implicit none
       integer(int32), intent(in), dimension(..) :: a
       integer(int32) :: r
     end function addv
     
     pure subroutine flatten(x, y)
       use iso_fortran_env
       implicit none
       integer(int32), intent(in), dimension(..) :: x
       integer(int32), intent(inout),  dimension(size(x)) :: y
     end subroutine flatten     
  end interface

  
! Main block start here
  a = 3
  print *, 'Function addv '

  write(*,'(a,i2)') 'Rank of a ',rank(a)
  write(*,'(a,4i2)')  'Shape of a ', shape(a)
  s = addv(a)  ! Call function addv which add all elements of a.
  write(*,'(ai4)') 'Sum vector ',s
  print *

  a = 3
  allocate(b(size(a)))
  write(*,'(a)')  'Subroutine flatten '
  write(*,'(ai4)') 'size of a ' ,size(a)
  write(*,'(ai4)') 'size of b ' ,size(b)
  write(*,'(a,i2,a,4i2)') 'rank of a  ',rank(a), ', shape of a ', shape(a)
  
  call flatten(a,b)  ! Collapse any n'th rank vector into a 1-d vector.

  write(*,'(ai2,a,4i2)')'rank of b  ',rank(b), ', shape of b ', shape(b)
  write(*,'(a,i4)')  'size of b ' ,size(b)
  write(*,'(a,5i4)')  'b(1:N:4) ', b(1::4)

  
! Get dimensions of array a.

  j=rank(a)       ! Number of ranks
  allocate(l(j))  ! Make space for dimensions
  write(*,'(a,i2)') 'Size of vecror l ',size(l) 
  l = shape(a)    ! Assign the different dimensions into vector l.
  do j=1,size(l)
     write(*,'(a,i2,a,i2)') 'Dimension size of dimension ',j,' in a ',l(j)
  end do
  deallocate(l)
end program assumedranktest
