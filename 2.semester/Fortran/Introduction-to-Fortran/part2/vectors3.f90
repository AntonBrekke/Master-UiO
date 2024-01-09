!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
! Remember in Fortran all arrays start at 1, as in science.
!
! Example of vector or indexed variables and implied do loops.
! 
! 
! 
! Written by Ole W. Saastad, UiO
! September 2023.
! 
!
program vectors
  use iso_fortran_env
  implicit none

  integer(int16) :: i
  integer(int16), parameter :: n=10
  real(real32), dimension(n) :: a
  real(real64), dimension(n,n) :: b=1.0_real64
  real(real64), dimension(n,n,n) :: c
  real(real64), dimension(n,n/2,n/4) :: d 

  real(real32), dimension(4) :: e = [1, 2, 3, 4]
  real(real64), dimension(n/2) :: f = [ (i, i=1,n/2)]
  real(real32), dimension(n/2) :: g = [ (log(real(i)), i=1,n/2)]
  real(real32), dimension(-5:5) :: h = 1.4142_real32
  complex(real32), dimension(5,5,5,5) :: w = (1.4_real32, 1.5_real32) ! Each element will have
                                                                      ! re=1.4 and im=1.5
  print *, 'Vectors sizes and dimensions'
  print *, 'size a: ', size(a)
  print *, 'size b: ', size(b)
  print *, 'size c: ', size(c)
  print *, 'size d: ', size(d)
  print *, 'size h: ', size(h)
  print *, 'size w: ', size(w)
  print *, 'shape a :', shape(a)
  print *, 'shape b :', shape(b)
  print *, 'shape c :', shape(c)
  print *, 'shape d :', shape(d)
  print *, 'shape h :', shape(h)
  print *, 'shape w :', shape(w)

! Here we want to print the vector f using implied do loop.
  print *,'f:'
  print '(f5.2)', (f(i), i=1,n/2) 

! And the vector g.  
  print *,'g:'
  print '(f5.2)', (g(i), i=1,n/2)

  print *,'g (every second element, vector syntax) :'
  print '(f5.2)', g(1:n/2:2)          ! The vector syntax allow step, here 2.

! The complex data type
  print *,'w (complex data type)'
  print '(x2(f5.2))', w(1,2,3,4)
  print '(xa,f5.2)', 'real part', w(4,3,2,1)%re
  print '(xa,f5.2)', 'imag part', w(4,3,2,1)%im
  print *, 'vector syntax'
  print '(x4(2(f5.1)",",f5.1))', w(1,1,1,:)

! A lot of vector, matrix intrinsic functions part for standard fortran.  
  print *,'Vector functions built in :'
  print *,'dot product of f and g ', dot_product(f,g)
  print '(a,x4(2(f5.1)",",f5.1))',' conjugate of w ', conjg(w(1,1,1,:)) ! one 5 elements.
  print *,'Matrix matrix multiplication, subset of b' ! We reduce b to 3 by 3 matrix.
  print '(2(f5.1,","),f5.1)',matmul(b(1:3,1:3), b(1:3,1:3))
end program vectors
