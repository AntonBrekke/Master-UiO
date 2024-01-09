C     
C «Introduction to Fortran»
C NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
C 
C Ole W. Saastad, University of Oslo  & NRIS
C 
C Mariner spacecraft bug from 1962.
C https://arstechnica.com/civis/threads/mariner-1-fortran-bug.862715/    
C
C Ole W. Saastad, UiO.
C September 2023
C
C
C  module load  GCC/12.2.0 to get 2008 standard
C
C The 12.2.0 GNU Fortran implements the Fortran 77, 90 and 95 standards completely, 
C most of the Fortran 2003 and 2008 standards, and some features from the 2018 standard. 
C
C
C      gfortran -std=legacy  mariner.f
C
C
      PROGRAM D15I                                                      CARD 1 comments after 
      DO 15 I=1.100                                                     CARD 2 column 72.
 15   CONTINUE                                                          CARD 3 Blanks are 
      PRINT *,DO15I                                                     CARD 4 insignificant
      END                                                               CARD 5 DO15I is a variable
