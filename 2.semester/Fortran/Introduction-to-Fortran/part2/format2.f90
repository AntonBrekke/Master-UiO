!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! 
!
! Example of advanced formats and internal files.
! 
! 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 
!
!
!   A - text string, characters
!   D - double precision numbers, exponent notation
!   E - real numbers, exponent notation
!   F - real numbers, fixed point format
!   G - numbers, scientific format
!   I - integer
!   X - horizontal skip (space)
!   / - vertical skip (newline)
!

program format2
  use iso_fortran_env
  implicit none

  integer(int16) :: j,k,n
  character(len=20) :: form
  
  write(*,'(a)') 'Dynamic formats, advanced formatting, internal file'
  write(*,'(a)', advance='no') 'How many numbers >'
  read(*,*) n

  write(form,'(a,i0,a)') '(',n,'(i1,x))'         ! This is called internal file.
  print *,'Format sting now looks like: ', form
  write(*,form) (j*j, j=1,n)                     ! Implied do loop.

! We need to adjust the width of the integer to fit the size.
  
  print *,log10(float(n*n))       ! How many numbers ?
  k=ceiling(log10(float(n*n+1)))  ! log10 of 100 is 2, need 3 for 100 hence +1,
                                  ! ceiling round up to neerest integer.

! We can now prepare the format string:
  
  write(form,'(a,i0,a,i0,a)') '(',n,'(i',k,',x))' 
  print *,'Format sting now looks like: ', form                                   

  write(*,form) (j*j, j=1,n)

! As 25+ (about) numbers would break the line, an exercise is to only
! emit 80 characters per line. 
!
! What happen when entering 1000 numbers ?
  
end program format2

  

  
