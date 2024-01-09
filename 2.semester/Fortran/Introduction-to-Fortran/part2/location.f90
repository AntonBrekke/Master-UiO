!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!
! Example derived data type.
! 
! 
! Written by Ole W. Saastad, UiO
! October 2023.
! 
program location

  type :: t_location
     real :: lat, lon
  end type t_location


  type(t_location) :: loc

  loc%lat=60.0
  loc%lon=11.0
  print *,'Your location is :', loc

  loc = t_location(20.0, 120.0)
  print *,'Your location is :', loc
  
  loc = t_location(lon=-30.0, lat=-80.0)
  print *,'Your location is :', loc

! Try to adapt the program to print out the location in ddmm.mm format with
! North/south and East/West indexing. West and South are negative.  

  
end program location
  
  
  
