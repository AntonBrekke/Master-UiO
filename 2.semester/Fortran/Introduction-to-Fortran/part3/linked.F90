
!
! Simple linked list demo in fortran.
! Written by Ole W. Saastad, UiO
! Dec. 2022.



! Compile with no debug information emitted :
! gfortran linked.F90
! Emit debug information :
! gfortran -DDEBUG linked.F90
!
! C preprocessor will automatic be run before compilation if extention
! .F or .F90 is used. If not set the -cpp to use preprocessor. 
!
! Alternative with DEBUG or without DEBUG information compiled in :
! cpp -DDEBUG  linked.F90  > linked.f90
! cpp linked.F90 > linked.f90
! Compile the f90 file.
! gfortran linked.f90
!

program linked
  use iso_fortran_env ! Include a set of constands functions. 
  implicit none       ! Make sure that all undeclared variables are errors.

! An object type with a variable and a poiter of the same type.
  type :: line
     real(real64) :: x
     type(line), pointer :: next
  end type line

! We only deal with pinters no targets.
  
  type(line), pointer :: head, current ! Declare two pointes if type line.
  integer :: j

  nullify(head, current) ! Make sure the pointer are point at null
  allocate(head)         ! Allocate an object, head is of type line
  current => head        ! Set current to point to object pointed to by head.
  do j=1, 5
     current%x = j          ! Assign an integer value to object's x variable.
#ifdef DEBUG
     print *, "currents's x ", current%x ! Write out the x value in the object 
                                         ! that current is pointing to.
#endif                           
     allocate(current%next) ! Allocate next line object.
     current => current%next! Move current to next object.
  end do
  
#ifdef DEBUG
  print *, "Head's x ", head%x
  print *, "linked list"
#endif
  
  current => head   ! Start with the first object in the chain.
  do
     if (.not. associated(current%next)) then ! Is next pointing to a                                                                         ! allocated object ?
#ifdef DEBUG
        print *, "exit"                       
#endif
        exit                                  ! if not exit the do loop,
     end if                                   ! and we're done.
#ifdef DEBUG
     print *, "currents's x ", current%x
#else     
     print *, current%x       ! Write out the object's x pointed to by current.
#endif
     current => current%next  ! Move current to the next object. 
  end do

! We just exit and let the OS clean up and deallocate. 
end program linked
