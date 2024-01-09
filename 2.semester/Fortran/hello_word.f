      ! Write "gfortran -o filename hello_word.f" to run 

      ! https://github.uio.no/olews/Introduction-to-Fortran
      
      program helloworld
      implicit none
      character*13 hello_string
      hello_string = "Hello, world!"
      write (*,*) hello_string
      end program helloworld

