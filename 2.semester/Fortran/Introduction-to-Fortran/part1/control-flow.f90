!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!
! Example of loop control flow and selecting
! 
!
! 
! Written by Ole W. Saastad, UiO
! February 2023.
! 
!
program control
  implicit none
! Using default sizes for all data types, being lazy.
  integer :: a=1,b=1,i,j
  integer, parameter :: one=1
  real :: T=273.15, C

  print *, "Loops and flow control, select statement"
  print *
  print *,"Infinite do loop, Cycle and exit"
  do 
     a=a+1
     if (a<5) then
        cycle
     end if
     if (a>10) then
        exit
     end if
     print *,a
  end do
  print *

  print *, "While loop"
  do while (b<3)
     print *,b
     b=b+1
  end do
  print *

  print *,"Temperature loop, real type loop count"
  do while (T < 300)
     C = T - 273.15
     T = T + 1.0
     if (mod(T,10.0) > 1.0) then
        cycle
     end if
     print '(1x,"Temperatures: ",f6.2," K, ",f4.1,"°C" )', T, C
  end do
  print *


  print *, "Select statement"
  do j=1,4
     select case (j)
        case(one)          ! Case must be a constant, or a parameter variable.
           print *, "One"
           i=1             ! Many lines of code are allowed.
        case(2)
           print *, "Two"
        case default
           print *, "Many"
     end select
  end do
  print *


! Loop count at fixed at start, nothing new, but important to remember if you are used to C.
  j=3
  print *, "Loop iterations are set at start"
  do i=1,j
     print *,i,j
     j=0    
  end do


end program control
