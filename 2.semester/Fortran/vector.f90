program vector 
    use iso_fortran_env
    ! implicit none 
    integer(int8), dimension(10) :: x
    integer(int8), dimension(5) :: y = (2)  ! Assigning all elements as 2 
    integer(int8), dimension(5) :: z = [1,2,3,4,5]  ! Assigning specific values
    integer(int16), parameter :: n=5
    real(real32), dimension(n) :: A = [(sin(REAL(i)), i=0, n-1)]  ! Assigning specific values
    x = 3       ! Assigning all elements as 3 
    print *, x
    print *, y
    print *, z
    print '(10F7.3)', A
end program vector 