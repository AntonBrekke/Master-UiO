

program lesinn


  type linje
     real :: x,y
     type(linje), pointer :: neste_linje
  end type linje
  
  integer :: ioresult
  integer :: ant_par,i=1
  real :: xi,yi
  real, dimension(:), allocatable :: xx,yy
  type(linje), pointer :: start, travers
  
  allocate(start)
  travers=>start
  nullify(travers%neste_linje)
  
  print *,'Allokert forste objekt'
  
  do	
     write(*,'(a)',advance='no') 'Gi 2 tall (end EOF char CTRL-D)>'
     read(*,*,iostat=ioresult) xi, yi
     
     if (ioresult > 0) then
        print *,'Bare tall godtas'
        cycle
     endif
     if (ioresult < 0) exit
     travers%x = xi
     travers%y = yi
     allocate(travers%neste_linje)
     travers=>travers%neste_linje
     ant_par = ant_par + 1
  enddo
  
  print *, 'EOF'
  
  nullify(travers%neste_linje)
  travers=>start
  
  print *,'Lest data inn og laget pekerkjede'
  
  do
     print *,travers%x, travers%y
     travers=>travers%neste_linje
     if (.not. associated(travers%neste_linje)) exit
  enddo
  
  allocate(xx(ant_par))
  allocate(yy(ant_par))
  
  travers=>start
  do	
     xx(i)=travers%x
     yy(i)=travers%y
     travers=>travers%neste_linje
     if (.not. associated(travers%neste_linje)) exit
     i=i+1
  enddo
  
  print *,'Fylt array'
  
  do i=1,ant_par
     print *,xx(i),yy(i)
     
  enddo
  
end program lesinn
