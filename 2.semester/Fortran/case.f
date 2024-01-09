      program case 
        use iso_fortran_env

        implicit none
        character :: c 

        write(*,'(a)', advance='no')'Give a character >'
        read(*,*) c

        select case(c)
          case('0':'9')
             print *, "Numbers!"
        end select
      end program case