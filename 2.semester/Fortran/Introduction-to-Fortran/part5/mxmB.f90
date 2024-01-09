!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Functions, first approach. 
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




program mx
  use timings
  implicit none
  real(real64), dimension(:,:), allocatable :: a, b
  real(real64), dimension(:,:), allocatable   :: c
  real(real64) :: alpha=1.0, beta=1.0
  integer(int32) :: i,j,l,n
  character(len=5) :: arg


  if (command_argument_count() < 1) then
     print *,'Usage : ./a.out SIZE'
     call exit(2)
  end if
  call get_command_argument(1, arg)
  read(arg,*) n
  allocate(a(n,n),b(n,n),c(n,n))

  call random_number(a)
  call random_number(b)

  call start_timer
! Call an optimised matix matrix library routine.   
  call  dgemm('n', 'n', n, n, n, alpha, a, n, b, n, beta, c, n)
  call stop_timer
  call showresults(n)

  write(*,'(a,2(g10.2,2x ))') 'Done c(1,1), sum(c)', c(1,1), sum(c)

  deallocate(a,b,c)
end program mx

