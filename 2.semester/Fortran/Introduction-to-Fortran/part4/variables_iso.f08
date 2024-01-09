!
!
! «This is not your Grandmother's fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
!
! Variables in fortran, sizes and declaration.
!
! Ole W. Saastad, UiO.
! February 2023
!
!
!  module load  GCC/12.2.0 to get 2008 standard
!
! The 12.2.0 GNU Fortran implements the Fortran 77, 90 and 95 standards completely, 
! most of the Fortran 2003 and 2008 standards, and some features from the 2018 standard. 
!
!
module data 
  implicit none
! Constants for data types, set up yourself. Here we use human approach.
! At least,  p='precision decimals' r='exponent range' 
! or number of figures for integers.
!
  integer, parameter :: r4  = selected_real_kind(p=6,r=20)  ! at least 6 decimal places
  integer, parameter :: r8  = selected_real_kind(p=15,r=307)! at least ± 307 exponent
  integer, parameter :: r16 = selected_real_kind(p=27,r=2400)
  integer, parameter ::  i1 = selected_int_kind(2) ! at lest 2 digits.  
  integer, parameter ::  i2 = selected_int_kind(3)
  integer, parameter ::  i4 = selected_int_kind(9)
  integer, parameter ::  i8 = selected_int_kind(17)
  integer, parameter :: i16 = selected_int_kind(33)

! Math constants
  real(r8), parameter :: pi8 = 4.0_r8 * atan(1.0_r8)  ! Constant 1.0 declared r8.
  real(r8), parameter :: e8 = exp(1.0_r8)
  real(r16),parameter :: pi16 = 4.0_r16 * atan(1.0_r16)
  real(r16),parameter :: e16 = exp(1.0_r16)
  complex(r4),parameter  :: j4  = (0,1_r4)
  complex(r8),parameter  :: j8  = (0,1_r8)
  complex(r16),parameter :: j16 = (0,1_r16)
end module data




program variables 
  use data
  use iso_fortran_env
!
! https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fC_005fBINDING.html#ISO_005fC_005fBINDING
! INT8, INT16, INT32, INT64
! REAL32, REAL64, REAL128
!
  use iso_c_binding
!
! https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fC_005fBINDING.html#ISO_005fC_005fBINDING
!
! Using the standard environment make it simpler, while still having control. However, the 
! user need to know about bit sizes and precision. The the 'selected_*' use a human approach
! using digits of precision and exponent range. 
!
  implicit none
  logical(i1)  :: ll1
  logical(i2)  :: ll2
  logical(i4)  :: ll4
  logical(i8)  :: ll8       ! Spending 8 bytes for a logical seems strange. 
  integer(int8)   :: ii1
  integer(int16)  :: ii2
  integer(int32)  :: ii4
  integer(int64)  :: ii8
  integer(i16)     :: ii16
  real(real32)     :: rr4
  real(c_float)    :: rrc4
  real(real64)     :: rr8
  real(c_double)   :: rrc8
  real(real128)    :: rr16
  complex(real32)  :: cc4
  complex(real64)  :: cc8
  complex(real128) :: cc16
  
  character(len=30)   :: form    


  print *,"Compiler version ; ", compiler_version()  ! constant from iso_env
  print *,"Compiler options : ", COMPILER_OPTIONS()

  print *,pi8
  print *,pi16
  print *,e8
  print *,e16
  print *,j4
  print *,j8


  form="(a,1x,i2,1x,a,1x,i3,1x,a)"

  write(*,fmt=form) "Sizeof l1      ", sizeof(ll1),"bytes,",sizeof(ll1)*8,"bits"
  write(*,fmt=form) "Sizeof l2      ", sizeof(ll2),"bytes,",sizeof(ll2)*8,"bits"
  write(*,fmt=form) "Sizeof l4      ", sizeof(ll4),"bytes,",sizeof(ll4)*8,"bits"
  write(*,fmt=form) "Sizeof l8      ", sizeof(ll8),"bytes,",sizeof(ll8)*8,"bits"
  write(*,fmt=form) "Sizeof i1      ", sizeof(ii1),"bytes,",sizeof(ii1)*8,"bits"
  write(*,fmt=form) "Sizeof i2      ", sizeof(ii2),"bytes,",sizeof(ii2)*8,"bits"
  write(*,fmt=form) "Sizeof i4      ", sizeof(ii4),"bytes,",sizeof(ii4)*8,"bits"
  write(*,fmt=form) "Sizeof i8      ", sizeof(ii8),"bytes,",sizeof(ii8)*8,"bits"
  write(*,fmt=form) "Sizeof i16     ", sizeof(ii16),"bytes,",sizeof(ii16)*8,"bits"
  write(*,fmt=form) "Sizeof real32  ", sizeof(rr4),"bytes,",sizeof(rr4)*8,"bits"
  write(*,fmt=form) "Sizeof c_float ", sizeof(rrc4),"bytes,",sizeof(rrc4)*8,"bits"
  write(*,fmt=form) "Sizeof real64  ", sizeof(rr8),"bytes,",sizeof(rr8)*8,"bits"
  write(*,fmt=form) "Sizeof c_double", sizeof(rrc8),"bytes,",sizeof(rrc8)*8,"bits"
  write(*,fmt=form) "Sizeof r16     ", sizeof(rr16),"bytes,",sizeof(rr16)*8,"bits"
  write(*,fmt=form) "Sizeof c4      ", sizeof(cc4),"bytes,",sizeof(cc4)*8,"bits"
  write(*,fmt=form) "Sizeof c8      ", sizeof(cc8),"bytes,",sizeof(cc8)*8,"bits"
  write(*,fmt=form) "Sizeof c16     ", sizeof(cc16),"bytes,",sizeof(cc16)*8,"bits"

  print *, "min/maxint i1 ", -huge(ii1)-1, huge(ii1) 
  print *, "min/maxint i2 ", -huge(ii2)-1, huge(ii2)
  print *, "min/maxint i4 ", -huge(ii4)-1, huge(ii4)
  print *, "min/maxint i8 ", -huge(ii8)-1, huge(ii8)
  print *, "min/maxint i16", -huge(ii16)-1,huge(ii16)


  print *, "min/max real r4 ",tiny(rr4),  huge(rr4)
  print *, "min/max real r8 ",tiny(rr8),  huge(rr8)
  print *, "min/max real r16",tiny(rr16), huge(rr16)
  print *, "Precision and range r4 ",precision(rr4), range(rr4)
  print *, "Precision and range r8 ",precision(rr8), range(rr8)
  print *, "Precision and range r16",precision(rr16), range(rr16)
   
  print *, "Complex parts, this is 2008 standard "
! Fortran 2008 and later revisions, %RE and %IM 
  print *, "Sizeof j4%re,  j4%im:  ",sizeof(j4%re),sizeof(j4%im)
  print *, "Sizeof j8%re,  j8%im:  ",sizeof(j8%im),sizeof(j8%im)
  print *, "Sizeof j16%re, j16%im: ",sizeof(j16%im),sizeof(j16%im)
  print *,j4%re, j4%im
  print *,j8%re, j8%im
  print *,j16%re, j16%im

  print *, "j**j give an interesting result: ", j4**j4
end program variables 
