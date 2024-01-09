!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Remember in Fortran all arrays start at 1, as in science.
!
! Example of namelist
! 
! 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
!

program namelistdemo
  use iso_fortran_env
  implicit none

  character(len=30) :: form
  integer(int16),dimension(2,2) :: a,b
  real(real32), dimension(2) :: x,y,z 
  character(len=20) :: land, sea, air, soil

  namelist /Mydata/ a, b, x, y, z, land, sea, air, soil
  form='(4(i2xi2x),3(f4.1x,f4.1x)/4a)'
  
  a=0 ; b=1
  x=[1.0, 2.0]; y=x*2 ; z=y*5
  land='Norway'
  sea='Northsea'
  air='dry'
  soil='clay'


! Write data to namelist file.
  
  open(unit=2, file='namelist.nml', status='unknown', action='write')
  write(2, nml=Mydata)
  close(2)
  print *,'Data written to file:'
  print form, a, b, x, y, z, land, sea, air, soil
  write(*,'(a)', advance='no') 'Hit enter to continue'; read(*,*)
  
  
  print *,'Clear the data:'
  a=0 ; b=0; x=[.0, .0]; y=0 ; z=0
  land='none'; sea='none'; air='none'; soil='none'
  print form, a, b, x, y, z, land, sea, air, soil

! Read data from namelist file.
  
  open(unit=2, file='namelist.nml', status='unknown', action='read')
  read(2, nml=Mydata)
  close(2)
  print *,'Data read from file:'
  print form, a, b, x, y, z, land, sea, air, soil
  write(*,'(a)', advance='no') 'Hit enter to continue'; read(*,*) 


  print *,'Clear the data:'
  a=0 ; b=0; x=[.0, .0]; y=0 ; z=0
  land='none'; sea='none'; air='none'; soil='none'
  print form, a, b, x, y, z, land, sea, air, soil

! Read data from a namelist file where lines are in random order.
! Demonstrate mapping of names.  
  
  print *,'Read nameslist with shuffeled lines'
  open(unit=2, file='namelist-shuffeled.nml', status='unknown', action='read')
  read(2, nml=Mydata)
  close(2)
  print *,'Data read from shuffeled file:'
  print form, a, b, x, y, z, land, sea, air, soil
  write(*,'(a)', advance='no') 'Hit enter to continue'; read(*,*)
  

  print *,'Clear the data:'
  a=0 ; b=0; x=[.0, .0]; y=0 ; z=0
  land='none'; sea='none'; air='none'; soil='none'
  print form, a, b, x, y, z, land, sea, air, soil

! Read data from a namelist file containing only a partial data set.
  
  print *,'Read nameslist with missing lines'
  open(unit=2, file='namelist-missing.nml', status='unknown', action='read')
  read(2, nml=Mydata)
  close(2)
  print *,'Data read from file:'
  print form, a, b, x, y, z, land, sea, air, soil
  
  
  
end program namelistdemo
  
   
