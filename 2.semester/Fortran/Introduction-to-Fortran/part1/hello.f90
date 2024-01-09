!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Start- your first Fortran program  
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


! First statement, this is the main (main as in python, C etc)

! The main routine start here.
program helloworld

! We need to include some modules, in this case the ISO environment
! which sets a lot of constants and variables.
!
!  https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fFORTRAN_005fENV.html
!
  use iso_fortran_env

! Implicit none is mandatory (while not for the program to work, but rather
! from keeping spaceships falling down). All undeclared variables are now of
! the type NONE. Eny assignment will flagged as an error by the compiler.  
  
  implicit none
  integer(int8) :: ioresult

! print statement write to the console, standard out (in Fortran this is
! number 6,  keyboard is 5. This differ from C.
  
  print *, 'Hello world'

! The '*' set the output to stanard out, e.g. console no. 6.  
  write(*,'(a)') 'Hello World'
  write(6,'(a)') 'Hello World' 

! Note leading space emitted by print.

! All io statements return a status, keyword is 'iostat'
  write(*,'(a)',iostat=ioresult) 'Hello World'
  print *, 'IOresult: ',ioresult

! Iostat is often more important when reading unpredicatble input
! from users.
! The 2018 standard https://j3-fortran.org/doc/year/10/10-007r1.pdf
! section 9.11.5 describe the messages 'iostat' return.
  
end program helloworld
