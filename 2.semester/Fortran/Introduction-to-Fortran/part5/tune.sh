#!/bin/bash


for FLAGS in \
    -O2 \
    -O3 \
    -Ofast \
    -fast \
    '-fast -fomit-frame-pointer' \
    '-Ofast -falign-loops' \
    '-Ofast -qopt-mem-layout-trans=4' \
    '-Ofast -qopt-prefetch=2' \
    '-Ofast -qopt-prefetch=5' \
    '-Ofast -unroll' \
    '-Ofast -unroll-aggressive' \
    '-Ofast -msse4.2' \
    '-Ofast -march=core-avx2' \
    '-Ofast -mtune=skylake' \
    '-Ofast -march=skylake' \
    '-Ofast -march=skylake-avx512' \
    '-Ofast -march=skylake-avx512 -fomit-frame-pointer' \
    '-Ofast -xCORE-AVX2' \
    '-Ofast -xCORE-AVX512' \
    '-Ofast -xCORE-AVX512 -fomit-frame-pointer' \
    '-Ofast -xCOMMON-AVX512' \
    '-Ofast -xCORE-AVX512 -falign-loops -mtune=skylake-avx512' \
    '-Ofast -xCORE-AVX512 -falign-loops -mtune=skylake-avx512 -fomit-frame-pointer' \
    '-Ofast -xCORE-AVX512 -falign-loops -mtune=skylake-avx512 -qopt-prefetch=2 -fomit-frame-pointer' \
    '-Ofast -xCORE-AVX512 -falign-loops -mtune=skylake-avx512 -qopt-prefetch=5 -fomit-frame-pointer' \
    '-Ofast -xCORE-AVX512 -falign-loops -mtune=skylake-avx512 -qopt-mem-layout-trans=3 -fomit-frame-pointer' \
    '-Ofast -xCORE-AVX512 -falign-loops -mtune=skylake-avx512 -qopt-prefetch=5 -qopt-mem-layout-trans=3 -fomit-frame-pointer' \
    '-Ofast -xCORE-AVX512 -falign-loops -mtune=skylake-avx512 -qopt-prefetch=1 -qopt-mem-layout-trans=3 -fomit-frame-pointer' \
    '-Ofast -xCORE-AVX512 -falign-loops -mtune=skylake-avx512 -qopt-prefetch=2   -qopt-mem-layout-trans=3 -fomit-frame-pointer' \
    '-Ofast -xCORE-AVX512 -falign-loops -mtune=skylake-avx512 -qopt-prefetch=3   -qopt-mem-layout-trans=3 -fomit-frame-pointer' \
    '-Ofast -xCORE-AVX512 -falign-loops -mtune=skylake-avx512 -qopt-prefetch=4   -qopt-mem-layout-trans=3 -fomit-frame-pointer' \
    '-Ofast -xCORE-AVX512 -falign-loops -mtune=skylake-avx512 -qopt-prefetch=5   -qopt-mem-layout-trans=3 -fomit-frame-pointer' \
    '-Ofast -xCORE-AVX512  -qopt-zmm-usage=high' \
    '-Ofast -xCORE-AVX512  -qopt-zmm-usage=low' \
    '-fast -march=skylake-avx512' \
    '-fast -march=skylake-avx512 -fomit-frame-pointer' \
    '-fast -march=skylake-avx512 -qopt-prefetch=2 -fomit-frame-pointer' \
    '-fast -march=skylake-avx512 -qopt-prefetch=5 -qopt-mem-layout-trans=3 -fomit-frame-pointer' \
    '-fast -march=skylake-avx512 -qopt-prefetch=2 -fomit-frame-pointer' \
    '-fast -march=skylake-avx512  -fomit-frame-pointer' \
    ; do 
        ifort $FLAGS mxm6.f90 timings.o mysecond.o 2>/dev/null; \
	echo -n $FLAGS  " " ;
#       objdump -d a.out | grep zmm | wc -l	   
	./a.out 4000 2>/dev/null | grep -v Done 
done
    
