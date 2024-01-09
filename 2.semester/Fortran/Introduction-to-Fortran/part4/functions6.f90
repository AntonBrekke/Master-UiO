!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Functions, recursive, calling itself.
!
! Ole W. Saastad, UiO.
! September 2023
!
!
!  module load  GCC/12.2.0 to get 2008 standard
!
! The 12.2.0 GNU Fortran implements the Fortran 77, 90 and 95 standards completely, 
! most of the Fortran 2003 and 2008 standards, and some features from the 2018 standard. 
!
!

! We declare an elemental function, that add element by element.
! 1,2 => 3 and [1,2]+[3,4] => 1+3, 2+4 => 4, 6  
! Similar to larger vectors.


elemental function addint(a,b) result(r)
  use iso_fortran_env
  implicit none
  integer(int32), intent(in) :: a, b
  integer(int32) :: r
  
  r = a + b
end function addint

! Elemental functions are independent and can be run in parallel.




program elemfunctest
  use iso_fortran_env
  implicit none
  integer(int32) :: j
  integer(int32), dimension(2,2) :: a
  integer(int32), dimension(2,2,2) :: b
  
! We need an interface here to inform that we use an elemental function.  
  interface
     elemental function addint(a,b) result(r)
       use iso_fortran_env
       implicit none
       integer(int32), intent(in) :: a, b
       integer(int32) :: r
    end function addint
  end interface

  print *, "Adding element by element "
  print *, "Scalars 1+2                   : ", addint(1, 2)
  print *, "2 element vector 1+3 & 2+4    : ", addint([1, 2], [3, 4])
  print *, "3 element vectors 1+4,2+5,3+6 : ", addint([1,2,3], [4,5,6])
  print *, "5 element vectors a+(a+a)     : ", addint([(j, j=1,5)], [(j+j, j=1,5)])
  a(1,:) = 1
  a(2,:) = 2
!  print *,a
  print *, "2x2 matrix [1,2,1,2]          : ", addint(a,a)
  
  b = reshape([(j, j=1,size(b))], [size(b,1),size(b,2),size(b,3)])
! We reshape a one dimensional vector [1,2,..,8] into a rank 3 array 2x2x2
  print *, "2x2x2 cube                    : ", addint(b,b)


end program elemfunctest
