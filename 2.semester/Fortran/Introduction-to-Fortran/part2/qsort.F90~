!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!  Vector syntax - part 1 pack and masking
!
! Quick sort demo.
!
! compile and run
! gfortran qsort.F90  && ./a.out
! gfortran -DDEBUG qsort.F90 && ./a.out
!
!
! 
!
!
!     Example belonging to "Modern Fortran in Practice" by Arjen Markus
!
!     This work is licensed under the Creative Commons Attribution 3.0 Unported License.
!     To view a copy of this license, visit http://creativecommons.org/licenses/by/3.0/
!     or send a letter to:
!     Creative Commons, 444 Castro Street, Suite 900, Mountain View, California, 94041, USA.
!
!     Compact implementation of the QuickSort algorithm
!
!     Note:
!     Because the function uses Fortran 90 features, its interface should be made
!     explicit when using it in an actual program. This is easiest via a module.
!
!
!
module qsort_functions
    implicit none
  contains
    recursive function qsort_reals( data ) result( sorted )
      real, dimension(:), intent(in) :: data
      real, dimension(1:size(data))  :: sorted

#ifdef DEBUG
      write(*,'(ai1a)') "****************************** ",size(data),&
            " ******************************************************"
      print *, "data ", data
      print *, "pack ",pack( data(2:), data(2:) > data(1) )
      print *, "qinp ", &
           [ pack( data(2:), data(2:) > data(1) ) , &
             data(1),                               &
             pack( data(2:), data(2:) <= data(1))   ]
#endif

      if (size(data) > 1) then
         sorted = &
              [ qsort_reals(pack( data(2:), data(2:) > data(1)) ), &
              data(1),                                           &
              qsort_reals(pack( data(2:), data(2:) <= data(1)) ) ]
      else
         sorted = data
      endif
      
    end function qsort_reals
end module qsort_functions



program qsorttest
  use qsort_functions

  integer, parameter :: N=5
  real, dimension(N) :: a,b
  integer :: i

! Comment assignment of a if you want random data.
  do i=1,N
     a(i)=rand()
  enddo
  a=[1,2,3,4,5]

  write(*,*) 'Order:'
  write(*,*) a
  print *, " "
  b=qsort_reals(a)
  write(*,*) 'Sorted:'
  write(*,*) b

end program qsorttest


  
