!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!
!
! Example of a simple class object
!
!
! 
! Written by Ole W. Saastad, UiO
! March 2022.
! 
! Compile gfortran simpleobject3.f90
! run ./a.out  
!
!
!

module calc
  use iso_fortran_env  
  implicit none
  private

! The variables here are attributes of the type, hence adressed 
! by <variable name>%<variable>, like in the main here st%x
! 
! Here we have public functions, some data with some data hidden.
     
  type, public :: stat
     real(real64), public, dimension(:), allocatable :: x
     real(real64), private :: avg, var, std, skew

     contains
       procedure, public :: calc => calculate
       procedure, public :: show => showstat
  end type stat

! This is definition of the procedures show above. 
contains
  subroutine calculate(this)
    implicit none
    class(stat), intent(inout) :: this
    real(real64) :: sum, var, sd, sk
    integer(int32) :: j
    
    sum=0.0
    do j=1,size(this%x)
       sum = sum + this%x(j)
    end do
    this%avg = sum/size(this%x)
    var=0.0; sk=0.0
    do j=1,size(this%x)
       var = (this%x(j) - this%avg)**2
       sk  = (this%x(j) - this%avg)**3
    end do
    var = var/size(this%x) ! This is not a sample, we have all data
    sd = sqrt(var)         ! If a sample we divide by N-1
    sk = sk/(size(this%x)*var**3)

    this%var = var
    this%std = sd   
    this%skew = sk
  end subroutine calculate

  subroutine showstat(this)
    implicit none
    class(stat), intent(inout) :: this

    write(*,'(a)') 'Statistics calcualted:'
    write(*,'(a,i3)') 'Number of elements : ', size(this%x)
    write(*,'(a,f6.2)') 'Smallest element : ', minval(this%x)
    write(*,'(a,f6.2)') 'Largest element  : ', maxval(this%x)
    if (size(this%x) > 4) then
       write(*,'(a,5f6.2)') 'First five elements : ', this%x(1:5)
    else
       write(*,'(a,5f6.2)') 'All       elements : ', this%x
    end if
    write(*,'(a,f6.2)') 'Mean     : ', this%avg
    write(*,'(a,f6.2)') 'Variance : ', this%var
    write(*,'(a,f6.2)') 'St. dev  : ', this%std
    write(*,'(a,f6.2)') 'Skew     : ', this%skew
  end subroutine showstat

end module calc


! Main start here, the program is equivalent to main in C.

program simpleobject
  use iso_fortran_env  
  use calc           ! This make variables and functions visible.
  implicit none
  
  type(stat) :: st   ! We declate the object. 

 
! Array x in class is declared allocatable, we need to allocate space.
  allocate(st%x(6))
  st%x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] ! Fill with 6 floats.

  
  print *, st%x ! Tha data itself are not declared private
  call st%calc
  call st%show

  deallocate(st%x)
  
end program simpleobject
