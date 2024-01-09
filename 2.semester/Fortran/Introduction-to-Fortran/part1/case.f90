!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Control case
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

program case
  use iso_fortran_env

  implicit none
  integer(int8)   :: j, i
  character :: c
  
  print *, "Select statement"
  do j=1, 10
     select case (j)
        case(1)            ! Case must be a constant, or a parameter variable.
           print *, "One"
           i=1             ! Many lines of code are allowed.
        case(2)
           print *, "Two"
        case default
           if (j > 4) exit ! Exit the do loop, not the case.    
           print *, "Many"
     end select
     print *,'End case loop'   
  end do
  print *

  write(*,'(a)', advance='no')'Give a character >'
  read(*,*) c
  select case(c)
     case ('a','e','i','o','u')
        print *,'Vovel'
     case ('x')
        print *,'Exit'
     case ('0':'9')
        print *, 'Numbers are not accepted !'
     case default
        print *,'Consonant (or special chars?)'
  end select
  print *

  write(*,'(a)', advance='no')'Give a character >'
  read(*,*) c
  select case (c)
  case ('a' : 'j')
     write(*,*)  'One of the first ten letters'
  case ('l' : 'p', 'u' : 'y')
     write(*,*)  'One of l, m, n, o, p, u, v, w, x, y'
  case ('z', 'q' : 't')
     write(*,*)  'One Of z, q, r, s, t'
  case default
     write(*,*)  'Other characters, which may not be letters'
  end select
  
  
end program case
