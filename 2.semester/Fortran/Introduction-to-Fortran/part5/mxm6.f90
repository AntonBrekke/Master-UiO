!
! «Introduction to Fortran»
! NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
! 
! Ole W. Saastad, University of Oslo  & NRIS
! 
!
! Ole W. Saastad, UiO.
! November 2023
!
! 
!
! Performance :
!
! GNU module load GCC/12.3.0 :
! gfortran -c -O3 -mavx512f -unroll  mxm6.f90 
! gfortran -c -Ofast -mavx512f  mxm6.f90 
! gfortran -c -Ofast -mavx512f -unroll  mxm6.f90 
! gfortran -c -march=skylake -Ofast -mavx512f -unroll mxm6.f90
!
!
! Intel module load intel/2023a : 
! ifort -c mxm6.f90
!
! Intel ifort uses the OpenMP backend for do concurrent, hence -qopenmp
! Set threads using : export OMP_NUM_THREADS=4
!
! ifort -c -qopenmp -O3 mxm6.f90 
! ifort -c -O3 -xcore-avx512 -qopenmp mxm6.f90
! ifort -c -O3 -xcommon-avx512 -qopenmp mxm6.f90
! 
!
!
! NVIDIA module load NVHPC/23.1-CUDA-12.0.0 :
! nvfortran -c -O3  mxm6.f90
!
! export ACC_NUM_CORES=2
! export CUDA_VISIBLE_DEVICES=3
!
! nvfortran -c -O3  -stdpar=multicore mxm6.f90 
! nvfortran -c -fast -Mvect=simd:512 -stdpar=multicore  mxm6.f90
!
! Offloading to accelerator
! nvfortran -c -O3  -stdpar=gpu mxm6.f90
! nvfortran -c -O3 -gpu=cc60 -stdpar=gpu mxm6.f90
! Link: <compiler>  -O3 -gpu=cc60 -stdpar=gpu mxm6.o timings.o mysecond.o
!
!
!
subroutine mxm(x,y,z)
  implicit none
  integer, parameter :: real64  = selected_real_kind(p=15,r=307)
  integer, parameter :: int32 = selected_int_kind(8)
  real(real64), dimension(:,:), intent(in)  :: x,y
  real(real64), dimension(:,:), intent(inout) :: z
  integer(int32) :: i,j,l,n 

  n=size(x,1)

! Fortran 2008/2018 introduced do concurrent  
  do concurrent(j = 1:n)
     do l = 1,n
        do i = 1,n
           z(i,j) = z(i,j) + x(i,l)*y(l,j)
        enddo
     enddo
  end do

end subroutine mxm




program mx
  use timings
  implicit none
  real(real64), dimension(:,:), allocatable :: a, b
  real(real64), dimension(:,:), allocatable   :: c
  integer(int32) :: i,j,l,n
  character(len=5) :: arg


  
  interface
     subroutine mxm(x,y,z) 
       implicit none
       integer, parameter :: real64  = selected_real_kind(p=15,r=307)
       integer, parameter :: int32 = selected_int_kind(8)
       real(real64), dimension(:,:), intent(in)  :: x,y
       real(real64), dimension(:,:), intent(inout) :: z
       integer(int32) :: i,j,l,n 
     end subroutine mxm
  end interface

  if (command_argument_count() < 1) then
     print *,'Usage : ./a.out SIZE'
     call exit(2)
  end if
  call get_command_argument(1, arg)
  read(arg,*) n
  allocate(a(n,n),b(n,n))
  allocate(c(n,n),source=0.0_real64)

  call random_number(a)
  call random_number(b)

  call start_timer
  call mxm(a,b,c)
  call stop_timer
  call showresults(n)

  write(*,'(a,2(g10.2,2x ))') 'Done c(1,1), sum(c)', c(1,1), sum(c)

  deallocate(a,b,c)
end program mx

