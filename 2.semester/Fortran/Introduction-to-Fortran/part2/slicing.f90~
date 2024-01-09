!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
!
!
!
! Vector syntax - Slicing of arrays, advanced vector syntax.
! 
!
! Ole W. Saastad, February 2023
!
!


program slicing
  use iso_fortran_env
  implicit none
  integer(int8), dimension(6,6) :: x,y,z=0
  integer(int8), dimension(2,2) :: a
  integer(int8) :: n=6, i,j,k

! manipulating x into a 6x6 matrix.
  x = reshape([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,&
       20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36],&
       [6,6])
  y=x

  print '(a)', "Start matrix "
  write(*,'(6(i3))') ((y(i,j), j=1,n), i=1,n)
  print *
  print '(a)', "A row "
  print '(6(i3))', y(1,:)
  print '(a)', "A column"
  print '(6(i3))', y(:,1)
  print '(a)', "The diagonal"
  print '(6(i3))', (y(i,i), i=1,n)
  print '(a)', "Every second element"
  print '(3(i3))', y(1:n:2,1:n:2)
  print '(a)', "Combine elements"
  print *, merge(y(1:6,1), z(1:6,1) , y(1:6,1)<5)


  print '(a)', "Make a an upper triangulart matrix "
  do j=2,n
     y(j, 1:j-1)=0
  end do
  write(*,'(6(i3))') ((y(i,j), j=1,n), i=1,n)
  
  print '(a)', "Lower triangual"
  y=x
  do j=1,n
     y(j, j+1:n)=0
  end do
  write(*,'(6(i3))') ((y(i,j), j=1,n), i=1,n)
  
  print '(a)', "Extract a small 2x2 submatrix"
  a = x(3:4, 3:4)
  write(*,'(2(i3))') ((a(i,j), j=1,2), i=1,2)

  print '(a)', "Take square root of every second element"
  print '(3(f6.2))', sqrt(float( x(1:n:2, 1:n:2) ))

  print '(a)', "Updating a submatrix"
  y=x
  y(3:4, 3:4) = 6 * x(3:4, 3:4)  
  write(*,'(6(i5))') ((y(i,j), j=1,n), i=1,n)


  print '(ai3)', "Sum       : ", sum(x(1:2,1:6:2))
  print '(ai3)', "Max value : ", maxval(x(1,:),1)
  print '(ai3)', "Min value : ", minval(x(:,1))
  print '(ai4)', "Product   : ", product(x(1,:), 1, x(1,:)<15 )


end program slicing

