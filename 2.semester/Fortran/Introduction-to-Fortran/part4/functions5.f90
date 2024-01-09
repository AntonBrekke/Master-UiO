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

! We declare a recursive function to calculate the next Fibonacci number.
! 

recursive function fib(n) result(f)
  use iso_fortran_env
  implicit none
  integer(int32), intent(in) :: n 
  integer(int32) :: f
  
  if (n <= 2) then
     f = n
  else
     f = fib(n-1) + fib(n-2)
  end if

end function fib



program fibfunctest
  use iso_fortran_env
  implicit none
  integer(int32) :: d,f
  integer(int32) :: j

! We need to know how to call the function, an interface is needed
  interface
     recursive function fib(n) result(f)
       use iso_fortran_env
       implicit none
       integer(int32), intent(in) :: n
       integer(int32) :: f
     end function fib
  end interface

    
  do j=1,30
     write(*,'(10i8)', advance='no') fib(j)
     if (mod(j,10)==0) print *
  end do

  write(*,'(a,f10.8)') 'The golden ratio is connected to this sequence &
       fib(n+1)/fib(n) : ', real(fib(31),real32) / real(fib(30),real32)

! https://en.wikipedia.org/wiki/Golden_ratio
! https://www.mathsisfun.com/numbers/golden-ratio.html
 
 
end program fibfunctest
