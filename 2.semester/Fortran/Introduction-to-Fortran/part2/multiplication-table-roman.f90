!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!
! Example of subroutines, strings, formatting, an advanced example to have fun
! with when you're borded. Ideal for procastination. 
! 
! 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 
! 
!
program multab_roman
  use iso_fortran_env
  implicit none

  integer(int16) :: i, j, n
  character(len=60) :: form
  character(len=10) :: dec2rom  ! We should have written an interface here.
  
  write(*,*) 'Multiplication table:'
  form='(10(i3,x)/)'                  ! Ten (integers width 3 + one space) and a newline.
  do i=1,10
     write(*,form,advance='no') (i*j, j=1,10)  ! This is an implied do loop.
  end do

! The multiplication table in Roman numbers, it's not simple to multiply IX by XI
! by paper an pencil using only Roman numericals. 

  print *,'Two do loops'
  do i=1,10
     do j=1,10
        write(*,'(a)',advance='no') dec2rom(i*j)
     end do
     write(*,*)
  end do

  
  ! Or with implied loops in one line, after 10 strings write emit a newline.
  print *,'Implied loops and format'
  write(*,'(10a)') ((dec2rom(i*j), i=1,10), j=1,10)
  
end program multab_roman




! We declare a function of type "charcter(len=10), with a decimal number as input.
! Functions and subroutines are explained later.
function dec2rom(n) result(res)
  use iso_fortran_env
  implicit none
  character(len=10) :: res ! The function result must have a type.  
  integer(int16), intent(in) :: n ! An input parameter only, cannot be updated here. 
  integer(int16) :: k, j
  integer(int16), dimension(13)  :: num=[1000,900,500,400,100,90,50,40,10,9,5,4,1]
  character(len=2), dimension(13):: &
       rom=["M ","CM","D ","CD","C ","XC","L ","XL","X ","IX","V ","IV","I "]
  character(len=20) :: tmp  
  
  k=n     ! Need to use a local variable as n is an immutable input parameter.
  tmp=''  ! Need to initialise each function call as declaration initialisation only
          ! happen the fist time the function is called.
  
  do j= 1,13
     do while(k >= num(j))
        k = k - num(j)
        tmp=trim(tmp)//rom(j)
     end do
  end do

  res=tmp
  return
end function dec2rom
  
