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
! The data class contain the data and functions to operate on the data. 
!
! 
! Written by Ole W. Saastad, UiO
! December 2022.
! 
! Compile: gfortran classdemo1.f90
!
! run ./a.out c  xy1.txt 
!
! 'c' is an argument for using chain of ojects to import the ascii file.
! 'xy.txt' is a text file with two numbers per line.
!
! Some reference to info:
! https://stevelionel.com/drfortran/2020/06/30/doctor-fortran-in-not-my-type/
!

module data            ! We keep data in this module, and routines to handle
  use iso_fortran_env  ! the data. 
  Implicit none 

  type, public :: dataobject 
       real(real64), dimension(:), allocatable :: x,y
       character(len=80) :: header

     contains
       procedure :: import  => data_import 
       procedure :: imports => data_import_simple
       procedure :: importl => data_import_list
       procedure :: importc => data_import_chain
       procedure :: print   => data_print
       procedure :: sum     => data_sum
       procedure :: write   => data_write
       procedure :: read    => data_read
       procedure :: cleanup => data_cleanup
  end type dataobject

contains ! Routines to manipulate the data in object.

! A simple routine to read the data, read the file two times, 
! first to count number of x,y pairs, secondly to fill x, y arrays.

  subroutine data_import(this, filename)
    class(dataobject), intent(inout) :: this
    character(len=*), intent(in) :: filename
    integer :: iost, res, fn=17, j=-1
    character(len=132) :: line

    open(unit=fn, file=filename, status='old', action='read')
! Check number of lines, no data stored 
    do while (.true.)
       read(fn, fmt="(a)", end=100) line  ! Jump labels are outdated, avoid.
       j=j+1
    end do
! number of data lines now known, use of labels is discouraged. 
100 allocate( this%x(j), this%y(j), stat=res) 
    if (res/=0) then
       print *,"Allocation failed"
       call exit(1) ! Terminate program.
    end if

    rewind(fn) ! Spool the tape back to the beginning
    j=1
    read(fn, fmt='(a)') this%header
    do while (.true.)
       read(fn,*,iostat=iost) this%x(j), this%y(j)
       if (iost/=0) then
          exit ! Exit do loop, EOF or error.
       end if
       j=j+1      
    end do
    close(fn)
  end subroutine data_import


! A smart routine that only read the file once, user fortran 2003
! standard to append elements to a vector. 

  subroutine data_import_simple(this, filename)
    class(dataobject), intent(inout) :: this
    character(len=*), intent(in) :: filename
    integer(int8) :: iost, res, fn=17 
    character(len=132) :: line
    real(real64) :: xx, yy

    allocate(this%x(0), this%y(0), stat=res)
    if (res/=0) then
       print *,"Allocation failed"
       call exit(1)
    end if
    open(unit=fn, file=filename, status='old', action='read', iostat&
         &=iost)  
    if (iost/=0) then
       write(*,'(a)') "File not found, exiting."
       call exit(1) ! Terminate program.
    end if
    read(fn, fmt='(a)') this%header
    do while (.true.)
       read(fn,*,iostat=iost) xx, yy
       this%x = [ this%x, xx ] ! This is fortran 2003 standard.
       this%y = [ this%y, yy ]
       if (iost/=0) then
          exit ! Exit do loop
       end if
    end do
    close(fn)
  end subroutine data_import_simple


! Can we use a linked list of objects ?
!
! A somewhat smarter routine that only read the file once. 
! Generate a linked list of allocated data objects while reading,
! allocate vectors and fill them from the linked objects.

  subroutine data_import_list(this, filename)
    class(dataobject), intent(inout) :: this
    character(len=*), intent(in) :: filename
    
    type :: line
       real(real64) :: x, y
       type(line), pointer :: next
    end type line

    type(line), pointer :: first, current, hold
    real(real64) :: xx, yy 
    real(real64), dimension(:), allocatable :: x, y
    integer(int32) :: fn=17, iost, res, j=0 ! Only 2e9 lines possible
    ! with int32. 

    nullify(first, current, hold)
    
    open(unit=fn, file=filename, status='old', action='read', iostat&
         &=iost)
    if (iost/=0) then
       write(*,'(a)') "File not found, exiting."
       call exit(1) ! Terminate program.
    end if

    read(fn, fmt='(a)') this%header
    do
       read(fn,*,iostat=iost) xx, yy
       if (iost/=0) then
          exit
       end if
       j=j+1
       allocate(current) ! allocate new object
       if (res/=0) then
          print *,"Allocation failed"
          call exit(1) ! Terminate program.
       end if
       current%x = xx
       current%y = yy
       current%next => first ! point to previous link
       first => current      ! update head pointer   
    end do
    current => first ! point to beginning of the list    

! We now have all the datapoints and can allocate the vectors   
    allocate(this%x(j),this%y(j), stat=res)
    if (res/=0) then
       print *,"Allocation failed"
       call exit(1) ! Terminate program.
    end if

! We now have a reversed linked list.
    do
       if (.not. associated(current)) then ! end of list reached 
          exit
       end if
       this%x(j)=current%x
       this%y(j)=current%y
       j=j-1
       hold => current
       current => current%next ! go the next link in the list
       deallocate(hold)
    end do
! The list should now be empty.
    if  (associated(first)) then
       print *,"objects in list still allocated"
    end if
  end subroutine data_import_list


! A somewhat smarter routine that only read the file once. 
! Generate a linked list of allocated data objects while reading,
! allocate vectors and fill them from the linked objects.

  subroutine data_import_chain(this, filename)
    class(dataobject), intent(inout) :: this
    character(len=*), intent(in) :: filename
  
    type :: line
       real(real64) :: xl, yl
       type(line), pointer :: next
    end type line

    type(line), pointer :: head, current, hold ! Declare pointers of
    ! type line.
    integer :: j=0, iost, res, fn=17

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
          call exit(1) ! Terminate program.
       end if
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
       this%x(j) = current%xl
       this%y(j) = current%yl
       hold => current
       current => current%next
       deallocate(hold)
       j=j+1
    end do

  end subroutine data_import_chain


  subroutine data_print(this)
    class(dataobject), intent(in) :: this
    integer :: j
    character(len=20) :: form='(f6.2xf6.2)'

    print *,"the 10 first and the 4 last lines"
    do j=1,10
       write(*,fmt=form) this%x(j), this%y(j)
    enddo
    print *,"--------------"
    do j=(size(this%x)-3), size(this%x)
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
    write(fn) this%header
    write(fn) this%x, this%y
    close(fn)
  end subroutine data_write


  subroutine data_read(this, filename)
    class(dataobject), intent(inout) :: this
    character(len=80) :: filename
    integer :: fn=2

    open(unit=fn, file=filename, form="unformatted", action='read')
    read(fn) this%header
    read(fn) this%x, this%y
    close(fn)
  end subroutine data_read


  subroutine data_cleanup(this)
    class(dataobject), intent(inout) :: this
    
    deallocate(this%x, this%y)
  end subroutine data_cleanup

end module data



! Main program to test the data module.
! Observe that there is no datatype info for data etc.
! All such information is encapsulated in the class, all
! functions and subroutines are generic. 

program classdemo
  use iso_fortran_env
  use data
  implicit none 

  type(dataobject) :: obj
  character(len=80) :: filename ! Binary data file
  character(len=2) :: arg
  logical :: present


! Some magic to interact with the user, select type of import routine.
! Name of ASCII input file could be added.
! Some checks for input and files, many more checks er needed to make
! it foolproof.

  write(*, fmt='(a)', advance='no') "Demo of classes in fortran - "
  if (command_argument_count() < 1) then
     print *,'Usage : ./a.out method {c,l,s,d (default)} filename.txt'
     call exit(2)
  end if
! print *, command_argument_count()
  call get_command_argument(1, arg) ! The the first argument, if present. 

  call get_command_argument(2, filename) ! Filename, if given.
  print *,'read file: ',filename

  if (index(filename, '.')==0) print *,'Please use a "." in the filename'
  
  inquire(file=filename, exist=present) ! Check if file exist.
  if (.not. present) then
     print *,"File not found, using default xy.txt" 
     filename="xy.txt" ! could make a loop and ask for a new name.
     inquire(file=filename, exist=present) ! Check if default file exist.
     if (.not. present) then
        print *,'Default input file, ',trim(filename),' not found, exiting.'
        call exit(2)
     end if
  end if

  select case (arg)
  case ('c', 'C')
     print *, "Object Chain"
     call obj%importc(filename) ! ASCII data file
  case ('l', 'L')
     print *,"Object list"
     call obj%importl(filename) ! ASCII data file
  case ('s', 'S')
     print *,"Smart, append vector, 2003 std."
     call obj%imports(filename) ! ASCII data file
  case default
     print *,"Simple"
     call obj%import(filename) ! ASCII data file
  end select


! Data is now imported with one of the routines selected. 
! We now write the binary data to file, read it back in and 
! display som ot of it.

  filename="class.dta"
  print *, obj%header 
  write(*,fmt='(a,i4,1x,a,f8.2)') "Size of x: ", size(obj%x), "Sum of&
       & x and y: ", obj%sum()
  call obj%print
  call obj%write(filename)
  obj%header=""; obj%x=0; obj%y=0 ! Clear data
  print *, 'Cleared data and read in binary file'
  call obj%read(filename)
  print *, obj%header
  call obj%print
  call obj%cleanup
  
end program classdemo
