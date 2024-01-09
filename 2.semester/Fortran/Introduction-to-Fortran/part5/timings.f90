!
!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
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

module timings
  use iso_fortran_env
  implicit none
  real(real64) :: starttime, endtime

! We need an interface to the mysec C routine. We know that it
! return a c_double. Assigning it to a fortran real64 should be ok. 
  interface
     function mysecond() result (s)
       use, intrinsic :: iso_c_binding, only: c_double
       implicit none
       real(c_double) :: s
     end function mysecond
  end interface
  

  
contains
  subroutine start_timer
    use iso_fortran_env
    implicit none
    starttime = mysecond()
  end subroutine start_timer

  subroutine stop_timer
    use iso_fortran_env
    implicit none
    endtime = mysecond()
  end subroutine stop_timer

  subroutine showresults(n)
    use iso_fortran_env
    implicit none
    integer(real32), intent(in) :: n
    integer(int64) :: footprint
    real(real64) :: operations
    real(real64) :: walltime

    operations = real(n,real64)**3 ! N³ operations. 

    footprint = (n*n*8_int64*3_int64)/(1024_int64*1024_int64) ! n² divided with 2^20 => MiB
    walltime = endtime - starttime  

    write(*,'(a,i5,a)', advance='no') 'Total footprint a,b & c ',footprint,' MiB,'
    write(*,'(a,f8.3,a)', advance='no') ' Walltime ', walltime, ' seconds, '
!    write(*,'(a,es12.4)', advance='no') 'Operations ', operations
    write(*,'(f8.3,a)') (operations/walltime)*1e-9, ' Gflops/s '
  end subroutine showresults
  
end module timings
     
