!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!
!
! Example of classes and linked lists in fortran
!
! A simple ascii files with one line of header then x, y data pairs 
! of data is read in and used to fill two arrays x and y. 
! Then a linked linst of a number of these objects are generated from 
! files given on the command line.
!
! The data class contain the data and functions to operate on the data. 
! 
! 
! Written by Ole W. Saastad, UiO
! January, November 2023
!
! 
! Compile gfortran classdemo.F90 
! Compile with debug  gfortran -DDEBUG classdemo.F90 
! Run : /a.out xy1.txt xy2.txt 
! or : ./a.out xy1.txt xy2.txt xy3.txt xy4.txt xy5.txt 
!
!

!************* Module definitions start here *********************
!*****************************************************************

module data           ! We keep data in this module, 
  use iso_fortran_env ! and routines to handle the data. 
  implicit none 

  type, public :: dataobject 
       real(real64), dimension(:), allocatable :: x,y
       character(len=80) :: header, name
       type(dataobject), pointer :: next

     contains
       procedure :: import  => data_import 
       procedure :: print   => data_print
       procedure :: sum     => data_sum
       procedure :: write   => data_write
       procedure :: read    => data_read
  end type dataobject

contains

  subroutine data_import(this, filename)
    class(dataobject), intent(inout) :: this
    character(len=*), intent(in) :: filename
  
    type :: line
       real(real64) :: xl, yl
       type(line), pointer :: next
    end type line

    type(line), pointer :: head, current, hold ! Declare pointers of
    ! type line.
    integer :: j, iost, res, fn=17
    
    j=0 ! This is needed, if set to 0 in the declaration it would
    ! keep its last
        ! value when this subroutine was entered next time.
    nullify(head, current, hold) 
    allocate(head) 
    current => head

    open(unit=fn, file=filename, status='old', action='read', iostat&
         &=iost)
    if (iost/=0) then
       write(*,'(a)') "File not found, exiting."
       call exit(1) ! Terminate program.
    end if
    read(fn, fmt='(a)') this%header 
    do 
       read(fn,*,iostat=iost) current%xl, current%yl 
       if (iost/=0) then 
          nullify(current%next)
          exit ! Exit the do loop           
       end if
       j=j+1 
       allocate(current%next, stat=res) 
       if (res/=0) then
          print *,"Allocation failed"
          call exit(1) ! Terminate program, 1 is «real» error.
       end if
#ifdef DEBUG
       print *, j, associated(current%next)
#endif
       current => current%next 
    end do
    close(fn)
! Size of arrays are now known.
! We can now allocate the vectors x and y.

    allocate(this%x(j), this%y(j), stat=res)
    if (res/=0) then
       print *,"Allocation failed"
       call exit(1) ! Terminate program.
    end if
    current => head
    
    j=1
    do 
       if (.not. associated(current%next)) then
          exit
       end if
#ifdef DEBUG
       print *,"loop", j, current%xl, current%yl, associated(current&
            &%next)
#endif       
       this%x(j) = current%xl
       this%y(j) = current%yl
       hold => current
       current => current%next
       deallocate(hold)
       j=j+1
    end do

  end subroutine data_import


  subroutine data_print(this)
    class(dataobject), intent(in) :: this
    integer :: j
    character(len=20) :: form='(f6.2xf6.2)'

    print *,"the 5 first and the 3 last lines"
    do j=1,5
       write(*,fmt=form) this%x(j), this%y(j)
    enddo
    print *,"--------------"
    do j=(size(this%x)-2), size(this%x)
       write(*,fmt=form) this%x(j), this%y(j)
    end do
  end subroutine data_print


  function data_sum(this) result(s)
    class(dataobject), intent(in) :: this
    real(real64) :: s
    
    s=sum(this%x)+sum(this%y)
  end function data_sum


  subroutine data_write(this, filename)
    class(dataobject), intent(in) :: this
    character(len=80) :: filename
    integer :: fn=2
    
    open(unit=fn, file=filename, form="unformatted", action='write')
    write(fn) this%header, this%name
    write(fn) this%x, this%y
    close(fn)
  end subroutine data_write


  subroutine data_read(this, filename)
    class(dataobject), intent(inout) :: this
    character(len=80) :: filename
    integer :: fn=2

    open(unit=fn, file=filename, form="unformatted", action='read')
    read(fn) this%header, this%name
    read(fn) this%x, this%y
    close(fn)
  end subroutine data_read



end module data

!************* Module definitions end here ***************
!*********************************************************





!************* Main program start here ******************
!********************************************************


! Main program to test the data module.
! Observe that there is no datatype info for data etc.
! All such information is encapsulated in the class, all
! functions and subroutines are generic. 

program classdemo
  use iso_fortran_env
  use data
  implicit none 

  type(dataobject), pointer  :: head, hold, traverse 
  character(len=80) :: filename, rfname 
  logical :: present
  integer :: j, pos, data_sets


! Some magic to interact with the user

! Make a loop to loop over all input files.
  
  write(*, fmt='(a)') "Demo of classes in fortran "  
#ifdef DEBUG
  print *, command_argument_count()  
#endif
  data_sets = command_argument_count()
  if (data_sets < 1) then
     print *,"Usage ./a.out  file1 file2 .... fileN"
     print *,"Need at least one input file, exiting"
     call exit(2) ! Exit code 2 if for «misuse»
  end if

  allocate(head)
  traverse => head

  do j=1, data_sets 
     call get_command_argument(j, filename) ! Filename, if given.
#ifdef DEBUG
     print *, filename
#endif

! An alternative is to check all files befort start. 
     inquire(file=filename, exist=present) ! Check if file exist.
     if (.not. present) then
        print *,"File not found ", trim(filename)," exiting!"
        call exit(2) ! Exit code 2 if for «misuse»
     end if


     call traverse%import(filename)
! Text format data are now imported 

#ifdef DEBUG
     print *, traverse%x
     print *, traverse%y
#endif
     print *, traverse%header
     write(*,fmt='(a,i4,1x,a,f8.2)') "Size of x: ", size(traverse%x),&
           "Sum of x and y: ", traverse%sum()

     call traverse%print

! Need to name the object and make a file name for a binary file 
! with data.
     pos = scan(filename, '.')-1
     traverse%name = filename(1:pos)
     rfname = filename(1:pos)//'.dta'
#ifdef DEBUG
     print *, traverse%name
     print *, rfname 
#endif
     call traverse%write(rfname)
     call traverse%read(rfname)
#ifdef DEBUG
     print *, traverse%header, traverse%name
#endif
! Allocate an object for the next data set, but not an extra if we're at the last.
     if (j < data_sets) then
        allocate(traverse%next)
        traverse => traverse%next
     else
        nullify(traverse%next)
     end if
     print *,"-"
  end do

  print *, "Run through the data objects:"
  traverse => head
  do
     write(*,'(a,2xa,1xf8.2)') trim(traverse%header), trim(traverse%name), traverse%sum()
     traverse => traverse%next
     if (.not. associated(traverse)) then
        exit
     end if
  end do


! Clean and deallocate the objects.
  traverse => head
  do j=1, data_sets
     hold => traverse
     traverse => traverse%next
     deallocate(hold)
  end do


#ifdef DEBUG
  print *, associated(head) 
  print *, associated(hold)
  print *, associated(traverse) 
#endif


end program classdemo
