!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!
!
! Example of a simple object
!
!
! 
! Written by Ole W. Saastad, UiO
! March 2022.
! 
! Compile gfortran simpleobject2.f90
! run ./a.out 
!
!
!

module calc
  use iso_fortran_env  
  implicit none
  
! The variables here are attributes of the type, hence adressed 
! by <variable name>%<variable>, like in the main here st%x

  type, public :: stat
     real(real64), dimension(:), allocatable :: x
     
     contains
       procedure :: mean => calcmean

  end type stat
 
! These functions and subroutines are available directly 
! in the calc module. No prefix needed.  
contains
  real(real64) function calcmean(this)
    implicit none
    class(stat), intent(inout) :: this
    real(real64) :: sum
    integer(int32) :: j
    
    sum=0.0              
    do j=1,size(this%x)
       sum = sum + this%x(j)   ! Could use the intrinsic sum(x)
    end do
    calcmean = sum/size(this%x)
  end function calcmean

end module calc


! Main start here, the program is equivalent to main in C.

program simpleobject
  use iso_fortran_env  
  use calc           ! This make variables and functions visible.
  implicit none
  
  type(stat) :: st

  allocate(st%x(5))
  st%x = [1.0, 2.0, 3.0, 4.0, 5.0]
  
  print *, st%x
  print *, st%mean()
end program simpleobject
