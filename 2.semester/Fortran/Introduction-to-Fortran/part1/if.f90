!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Control flow, if-then-else
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

program control_flow
  use iso_fortran_env

  implicit none
  integer(int8)   :: a, b=10 
  logical :: p,q
  
  write(*,fmt='(a)',advance='no') 'Give an integer >'
! No test if an actual integer is given, be nice,
! all integers are signed, no unsigned. 
  read(*,*) a
  if (a < b) then
     write(*,fmt='(i3,a,i3)') a,' is less than ',b
  else
     write(*,fmt='(i3,a,i3)') a,' is greater than ',b    
  end if

! A single statement is possible on a single line.
  if (a==b) write(*,'(a)') 'They are equal'

! A logical can be assigned from outcome of a test. 
  p=(a==B)
  q=(a<b)
  print *,'p=(a == b) is:',p,'; q=(a < b) is:',q

! The test need parentesis, even here :
  if (p) then
     print *,'p is True'
  else
     print *,'p is False'
  endif

  print *,'p=',p,'q=',q
  
! Examples of logical variables and logical operators.
  if (p .and. q)        print *,'p and q true'
  if (p .or. q)         print *,'p or q true'
  if (p .eqv. q)        print *,'p xor q true'   
  if (.not. (p .or. q)) print *,'not (p or q) true'

  
end program control_flow


