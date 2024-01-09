/*

Mysecond with nanosec resolution.


Written by Ole W. Saastad, UiO 

gcc -c mysecond.c 

Fortran expect a training underscore after symbol name.
Hence addition of a weak symbol.

*/ 

/*
CLOCK_MONOTONIC : 
For this clock, the value returned by clock_gettime() represents the
amount of time (in seconds and nanoseconds) since an unspecified point
in the past (for example, system start-up time, or the Epoch). This
point does not change after system start-up time. The value of the
CLOCK_MONOTONIC clock cannot be set via clock_settime().

The CLOCK_MONOTONIC_RAW is unaffected by NTP 
*/

#include <time.h>

#pragma weak mysecond_ = mysecond

double mysecond()
{
  struct timespec tp;
  clock_gettime(CLOCK_MONOTONIC_RAW, &tp);
  return (double)tp.tv_sec + (double)tp.tv_nsec*1.0e-9 ;
}

