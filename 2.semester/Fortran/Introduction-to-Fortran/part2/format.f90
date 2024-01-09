!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Remember in Fortran all arrays start at 1, as in science.
!
! Example of vector or indexed variables and implied do loops.
! 
! 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 
!
!
!   A - text string, characters
!   E - real numbers, exponent notation
!   EN - real numbers engineering format
!   ES - real numbers scientific format
!   F - real numbers, fixed point format
!   G - numbers, scientific format, self modifying decimals or exponent.
!   I - integer
!   X - horizontal skip (space)
!   / - vertical skip (newline)
!

program format
  use iso_fortran_env
  implicit none

  integer(int8) :: i,j
  character(len=60) :: form
  integer(int8) :: k=123
  real(real32)  :: x=3.14

  print '(a)', 'Hello world'   ! Inline format
  form='(a)'                   ! One character or a string
  print form,'Hello world'     ! Format as variable
  write(*,form) 'Hello world'  

  
  form='(a/a)' ! One string a newline and another string (or character).
  write(*,form) 'Hello', 'world'  

  form='(3alxl)'
  write(*,form) 'Logical (format',trim(form),' ): ', .true., .false.  

  form='(ai6.4)'
  write(*,*) 'Format: ', form
  write(*,form) 'Leading zeros: ',-24

  
  form='(a,2x,i3,i3,f4.2)' ! One string, two spaces, one integer, one integer, one real.
  write(*,*) 'Format: ', form
  write(*,form) 'Numbers',k,k,x

  
  form='(a,2x,i3,x,i3,x,f4.2,x,e8.2)'  ! One string, 2 spaces, one integer, one space,
  write(*,*) form                      ! one integer, one space, one real, one real 
  write(*,form) 'Numbers',k,k,x,x      

  
  form='(a,2x,i3,x,i3,x,f4.2,x,e8.2,x,g10.3)' ! One string, 2 spaces, one integer, one space,
  write(*,*) 'Format: ', form                 ! one integer, one space, one real, one real,
  write(*,form) 'Numbers',k,k,x,x,x           ! one real
                                              
  
  x=x*1e-7
  write(*,*) 'Format: ', form
  write(*,form) 'Numbers',k,k,x,x,x           ! Same as above, but g behaves differently.

  
  form='(a, "," i3 "," i3 "," f12.9 "," e8.2 "," g10.3)' ! string, comma, integer, comma,
  write(*,*) 'Format: ', form                            ! integer, comma, float, comma, 
  write(*,form) 'Numbers',k,k,x,x,x                      ! float, comma, float
                                                         

! The 'en' (engineering format) and 'es' (scientific format).
  form='(a, ", " i3 ", " i3 ", " f12.9 ", " en10.2 ", " es10.2)'
  write(*,*) 'Format: ', form
  write(*,form) 'Numbers',k,k,x,x,x 
end program format

  

  
