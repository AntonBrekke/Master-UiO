C
C
C «Introduction to Fortran»
C NRIS (Norwegian Research Infrastructure Service), https://www.sigma2.no/nris
C 
C Ole W. Saastad, University of Oslo  & NRIS
C 
C Control case
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
C Must be compiled as legacy code : gfortran -legacy gotos.f
C
C This code demonstrate unstructured jumps, relic from assembly language
C days when only jumps were possibe      
C
      PROGRAM BAD_SYNTAX                                                CARD  1
      I=0                                                               CARD  2
 10   I=I+1                                                             CARD  3
      IF (I .GT. 10) GOTO 20                                            CARD  4
      PRINT *,I                                                         CARD  5
      GOTO 10                                                           CARD  6
 20   PRINT *,'DONE'                                                    CARD  7
C                                                                       CARD  8
C  COMPUTED GOTO                                                        CARD  9
 100  FORMAT(/'GIVE AN INTEGER'/)                                       CARD 10
      WRITE(*,100)                                                      CARD 11 
      READ(*,*) I                                                       CARD 12
      GOTO (30,40,50) I                                                 CARD 13
      PRINT *,'ONLY 1, 2 OR 3'                                          CARD 14 
      GOTO 90                                                           CARD 15
 30   PRINT *,'ONE'                                                     CARD 16
      GOTO 90                                                           CARD 17
 40   PRINT *,'TWO'                                                     CARD 18
      GOTO 90                                                           CARD 19
 50   PRINT *,'THREE'                                                   CARD 20
 90   CONTINUE                                                          CARD 21
C                                                                       CARD 22
C ARITMETRIC GOTO                                                       CARD 23
      IF (I) 60,70,80                                                   CARD 24
 60   PRINT *,'NEGATIVE',I                                              CARD 25
      GOTO 91                                                           CARD 26
 70   PRINT *,'NULL',I                                                  CARD 27
      GOTO 91                                                           CARD 28
 80   PRINT *,'POSITIVE',I                                              CARD 29
 91   CONTINUE                                                          CARD 30
      END
C This code is a VERY bad code !
