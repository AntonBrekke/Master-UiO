program do_loop
  use iso_fortran_env

  implicit none
  integer(int8)   :: i, n 
  integer(int8)   :: ioresult
  character(len=40) :: iomessage

  n=10
! Simple do loop
  print *,'Simple do loop:'
  do i=0, 8, 2               ! for i=1 to 8 step 2.
    print *, i
  end do

  print *,'Simple do loop2:'
  do i=0, n, n/2       
    print *, i
  end do
end program do_loop