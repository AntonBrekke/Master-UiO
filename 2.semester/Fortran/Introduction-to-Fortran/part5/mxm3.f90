!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
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

subroutine mxm(x,y,z)
  implicit none
  integer, parameter :: real64  = selected_real_kind(p=15,r=307)
  integer, parameter :: int32 = selected_int_kind(8)
  real(real64), dimension(:,:), intent(in)  :: x,y
  real(real64), dimension(:,:), intent(inout) :: z
  integer(int32) :: i,j,l,n 

  n=size(x,1)
  do j = 1,n
     do l = 1,n
        do i = 1,n
           z(i,j) = z(i,j) + x(i,l)*y(l,j)
        enddo
     enddo
  enddo

end subroutine mxm




program mx
  use timings
  implicit none
  real(real64), dimension(:,:), allocatable :: a, b
  real(real64), dimension(:,:), allocatable   :: c
  integer(int32) :: i,j,l,n
  character(len=5) :: arg


  
  interface
     subroutine mxm(x,y,z) 
       implicit none
       integer, parameter :: real64  = selected_real_kind(p=15,r=307)
       integer, parameter :: int32 = selected_int_kind(8)
       real(real64), dimension(:,:), intent(in)  :: x,y
       real(real64), dimension(:,:), intent(inout) :: z
       integer(int32) :: i,j,l,n 
     end subroutine mxm
  end interface

  if (command_argument_count() < 1) then
     print *,'Usage : ./a.out SIZE'
     call exit(2)
  end if
  call get_command_argument(1, arg)
  read(arg,*) n
  allocate(a(n,n),b(n,n))
  allocate(c(n,n),source=0.0_real64)
  
  call random_number(a)
  call random_number(b)

  call start_timer
  call mxm(a,b,c)
  call stop_timer
  call showresults(n)

  write(*,'(a,2(g10.2,2x ))') 'Done c(1,1), sum(c)', c(1,1), sum(c)

  deallocate(a,b,c)
end program mx

