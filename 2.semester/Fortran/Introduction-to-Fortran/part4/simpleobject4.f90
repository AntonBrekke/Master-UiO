!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!
!
! Example of a simple class
!
!
! 
! Written by Ole W. Saastad, UiO
! March 2022.
! 
! Compile gfortran simpleobject4.f90
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
     real(real64), private, dimension(:), allocatable :: x
     real(real64), private :: sum, avg, var, std, skew

     contains
       procedure, public :: gen =>  generate
       procedure, public :: calc => calculate
       procedure, public :: show => showstat
  end type stat

! This is definition of the procedures show above. 
contains
  subroutine generate(this, m) ! m is  optional.
    implicit none
    class(stat), intent(inout) :: this
    integer(int32), intent(in), optional :: m
    integer(int32) :: j, n=10 ! Default if no m given.
    
    if (present(m)) n=m 

    allocate(this%x(n))
    call random_seed()
    call random_number(this%x)
    this%x = 10.0 * this%x            
  end subroutine generate
    
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
    this%sum = sum
  end subroutine calculate

  subroutine showstat(this)
    implicit none
    class(stat), intent(inout) :: this

    write(*,'(a)') 'Statistics calcualted:'
    write(*,'(a,i4)') 'Number of elements : ', size(this%x)
    write(*,'(a,f8.5)') 'Smallest element : ', minval(this%x)
    write(*,'(a,f8.5)') 'Largest element  : ', maxval(this%x)
    if (size(this%x) > 4) then
       write(*,'(a,5f7.3)') 'First five elements : ', this%x(1:5)
    else
       write(*,'(a,5f7.3)') 'All       elements : ', this%x
    end if
    write(*,'(a,g10.3)') 'sum      : ', this%sum
    write(*,'(a,g10.3)') 'Mean     : ', this%avg
    write(*,'(a,g10.3)') 'Variance : ', this%var
    write(*,'(a,g10.3)') 'St. dev  : ', this%std
    write(*,'(a,g10.3)') 'Skew     : ', this%skew
  end subroutine showstat

end module calc


! Main start here, the program is equivalent to main in C.

program simpleobject
  use iso_fortran_env  
  use calc           ! This make variables and functions visible.
  implicit none
  
  type(stat) :: st   ! We declate the object. 

! Note that in the main program there are no type declarations nor any
! information about data types. All this is encapsulated in the object. 
  
  call st%gen(15) ! The argument is optional, defaults to 10.
  call st%calc
  call st%show
  
end program simpleobject
