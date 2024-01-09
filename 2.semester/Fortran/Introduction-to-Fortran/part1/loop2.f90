!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Control flow, loops
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

program control_flow2
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

  
  
! Loops with finer control
  do i=1, n                   ! for i=1 to n, step 1 (default)
     print *,'Count ',i
     if (i<3) cycle
     print *,'i Greater than 3'
     if (i>5) exit
  end do

  i=0
  print *,'Do while loop:'
  do while (i<10)
     i=i+1
     if (i<5) cycle
     print *,i
     if (i>7) exit
  end do


! Do while loop
  
  do while (.true.) ! This is an infinite loop.
     write(*,'(a)', advance='no') 'Give an integer >'
     
! All io operations return a status, the parameter is named 'iostat'.
     read(*,*, iostat=ioresult) i

! Exit loop when accepted result received or emit information.
     if (ioresult .eq. 0) then
        exit
     end if
  end do  
  print *,'An accepted integer: ',i


  
  ioresult=1
! Using the iostat as while loop test.  
  do while (ioresult .ne. 0)
     write(*,'(a)', advance='no') 'Give an integer >'
     read(*,*, iostat=ioresult, iomsg=iomessage) i
     if (ioresult /= 0) print *,iomessage
  end do
  print *,'An accepted integer: ',i

  
  
end program control_flow2
