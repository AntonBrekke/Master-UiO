      program helloworld 
        use iso_fortran_env
        implicit none 
        integer(kind=int8) :: ioresult
        print *, "Hello, World!"
        write(*, "(a)")
        write(6, "(a)")
      end program helloworld