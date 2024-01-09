!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
! Vector syntax - part 1 pack, unpack, masking, shifting and rotating
!
!
! Ole W. Saastad, UiO
! February 2023
!
!
! The square bracket form was added to the language in Fortran 2003. 
! 
program pakk
  implicit none
  integer :: i, j
  integer, dimension(6) :: v1
  integer, dimension(4) :: v2
  integer, dimension(:), allocatable :: v3 ! size of v3 is unknown at compile time.
  integer, dimension(2,2) :: m1=0
  logical, dimension(6) :: mask
  logical, dimension(2,2) :: mask22=.true.
  character(len=10) :: form
  v1 = [ 1, 0, 3, 0, 6, 0 ] ! Fortran 2003 syntax.
  v2 = [ 1, 0, 0, 2 ]
  
  m1(1,1)=1; m1(2,2)=1   ! make it an identiy matrix.

  mask =  v1 /= 0  ! equivalent: do j=1,size(v1) 
                     ! if (v1(j)/=0) then mask(j)=.true. else mack(j)=.false. end if
  form="(a,6I4)"
  
  print '(a)', "Packing, unpacking, shifting, rotating "
  print'(a)', "***************************** pack **************************************"
  write(*,form) "v1 :                                ", v1 
  write(*,'(a,6L4)') "mask v1 /= 0 :                      ",mask
  write(*,form) "pack(v1, mask):                     ",pack(v1, mask)
  write(*,form) "pack(v1, v1 /= 0) :                 ",pack(v1, v1 /= 0)
  write(*,form) "pack(v1, v1 > 1) :                  ",pack(v1, v1 > 1)
  print '(a,i2)',"size of pack(v1, v1 > 1) : ",size(pack(v1, v1 > 1))
! Note that pack reduces the size of the resulting vector.

  print '(a)', "v3 = pack(v1, v1 > 1))"
! Size of v3 unknown, at compile time, need to declate it as allocatable.
  allocate(v3(size(pack(v1, v1 > 1))))
  v3=pack(v1, v1 > 1)
  print '(a,2i3,a,i3)', "v3 : ", v3, ", size(v3): ", size(v3)
  deallocate(v3)

  print '(a)',"------------------------------------------------------------------------"
  write(*,form) "v2 :                                ", v2
  write(*,form) "pack(v2, v2 /= 0) :                 ",pack(v2, v2 /= 0)
  write(*,form) "pack(v2, v2 /= 0, [ 5, 6, 3, 4 ]) : ",pack(v2, v2 /= 0, [ 5, 6, 3, 4 ])

  print '(a)',"------------------------------------------------------------------------"
  write(*,form) "v1 :                                    ", v1 
  write(*,form) "pack(v1, v1(1:) > v1(1) ) :             ",pack(v1, v1(1:) > v1(1) )
  write(*,form) "pack(v1, v1(1:) <= v1(1) ) :            ",pack(v1, v1(1:) <= v1(1) )

  write(*,form) "pack(v1, v1 == 0, [4, 5, 6, 7, 8, 9]) : ",&
       pack(v1, v1 == 0, [4, 5, 6, 7, 8, 9])

  print '(a)',"------------------------------------------------------------------------"
  write(*,form) "v1 :                                    ", v1 
  print '(a,6L4)', "mask:                                   ", mask
  write(*,form) "pack(v1, mask, [4, 5, 6, 7, 8, 9]) :    ",&
       pack(v1, mask, [4, 5, 6, 7, 8, 9])

  print '(a)',"------------------------------------------------------------------------"
  write(*,'(a)') "Matrix(2x2) m1 to vector(1x4):"
  write(*,'(2i2)') ((m1(i,j), j=1,2), i=1,2)
  write(*,'(a,4i2)') "pack(m1): ", pack(m1,.true.)
  print *

  write(*,'(a)') "******************************* unpack **********************************"
  write(*,form) "v1 :                                          ", v1 
  write(*,'(a,6L4)') "mask v1 /= 0 :                                ",mask
  write(*,form) "unpack(v1, mask , [-1, -2, -3, -4, -5, -6]) : ",&
       unpack(v1, mask , [-1, -2, -3, -4, -5, -6])
  print '(a)',"------------------------------------------------------------------------"  

  print '(a)', "matrix(2x2) to vector(1x4)"
  write(*,form) "v2 : ", v2
  m1 = unpack(v2, mask22, 0) ! mask22 is 2x2 all elements = .true. 
  print '(a)', "unpack(v2, <2x2 truemask>, 0)" 
  write(*,'(2i2)') ((m1(i,j), j=1,2), i=1,2)

  print '(a)', "******************************* cshift **********************************"
  v1 = [1,2,3,4,5,6]
  print form, "v1 :             ", v1
  print form, "cshift(v1, 1) :  ", cshift(v1, 1) 
  print form, "cshift(v1, -1) : ", cshift(v1, -1)
  print form, "cshift(v1, 4) :  ", cshift(v1, 4)
  print '(a)',"------------------------------------------------------------------------"

  print '(a)', "m1 :"
  write(*,'(2i2)') ((m1(i,j), j=1,2), i=1,2)
  print '(a)','cshift(m1, [1, 1], 2) :'
  m1 = cshift(m1, [1, 1], 2)
  write(*,'(2i2)') ((m1(i,j), j=1,2), i=1,2)

  print '(a)', "******************************* eoshift *********************************"
  print form, "eoshift(v1, 2, -8) :  ", eoshift(v1, 2, -8)
  print form, "eoshift(v1, -2, -8) : ", eoshift(v1, -2, -8)
  print *

  print form, "******************************* loc *************************************"
  v1 = cshift(v1, 2)
  print form, "v1 :                      ", v1
  print form, "maxloc(v1, v1 < 6) :      ", maxloc(v1, v1 < 6)
  print form, "v1(maxloc(v1, v1 < 6) ) : ", v1(maxloc(v1, v1 < 6) )
  print *
end program pakk
