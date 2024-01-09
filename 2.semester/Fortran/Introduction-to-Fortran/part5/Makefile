#
# «Introduction to Fortran»
# NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
# 
# Ole W. Saastad, University of Oslo  & NRIS
# 
#
# Ole W. Saastad, UiO.
# November 2023
#
# 
#


# C Compiler:
CC = gcc
# Fortran compiler (be sure to select the correct compiler) :
#FC = gfortran
FC = nvfortran
#FC = ifort


# Some opt flags :
# gfortran : -O3 -mavx2  -mavx512f 
# nvfortran : -O3 -fast -Mvect=simd:256 -Mprefetch -Munroll 
# ifort : -O3 -Ofast -xCORE-AVX2 -march=skylake-avx512 -fomit-frame-pointer

# Options (optimisation flags differ from compiler to compiler) :
OPT = -O3  
DEBUG =  -g

# Targets :
mxm1:	timings.o timings.mod mysecond.o 
	$(FC) $(DEBUG) $(OPT)  -o mxm1.x  mxm1.f90 timings.o mysecond.o

mxm2:	timings.o timings.mod mysecond.o 
	$(FC) $(DEBUG) $(OPT)  -o mxm2.x  mxm2.f90 timings.o mysecond.o

mxm3:	timings.o timings.mod mysecond.o 
	$(FC) $(DEBUG) $(OPT)  -o mxm3.x  mxm3.f90 timings.o mysecond.o

mxm4:	timings.o timings.mod mysecond.o 
	$(FC) $(DEBUG) $(OPT)  -o mxm4.x  mxm4.f90 timings.o mysecond.o

mxm5:	timings.o timings.mod mysecond.o 
	$(FC) $(DEBUG) $(OPT)  -o mxm5.x  mxm5.f90 timings.o mysecond.o

mxm6c:	timings.o timings.mod mysecond.o
	$(FC) $(DEBUG) -O3 -stdpar=multicore -o mxm6c.x  mxm6.f90 timings.o mysecond.o	

mxm6a:	timings.o timings.mod mysecond.o 
	$(FC) $(DEBUG) -O3 -gpu=cc60 -stdpar=gpu -o mxm6a.x  mxm6.f90 timings.o mysecond.o

mxm-library: timings.o timings.mod mysecond.o  
	$(FC) $(DEBUG) $(OPT)  -o mxm-library.x mxm-library.f90 -qmkl=sequential \
	timings.o mysecond.o
mxm1-profile: timings.o timings.mod mysecond.o 
	$(FC) $(DEBUG) $(OPT) -pg -fprofile-arcs -ftest-coverage -o mxm1.x mxm1.f90  timings.o mysecond.o


# Support utilities :
timings.o timings.mod: timings.f90 
	$(FC) -c timings.f90

mysecond.o: mysecond.c
	$(CC) -c mysecond.c


# Clean up command (phony used to inform that clean is not a real file target).
.PHONY:	clean
clean:
	rm -f *.o a.out mxm*.x *.mod
