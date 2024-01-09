      program decision 
      logical :: p=.true. 

      if (p) then print *, "True"
      if (a==b) write(*, '(a)') "They are equal"
      else (a=0)
      end if 
      end program decision