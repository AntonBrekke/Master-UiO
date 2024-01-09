


module data

  integer, parameter, public :: real8 = selected_real_kind(15,307)
 
  real(kind=real8), dimension(:), pointer, public :: x, xs, y, ys, y_calc

end module data


module innut

  public :: lesinn
  integer, parameter, private :: real8 = selected_real_kind(15,307)

    type, public :: linje
       real(kind=real8)  :: x, xs, y, ys
       type(linje), pointer :: neste_linje
    end type linje


 
  contains

    subroutine lesinn(xx,xxs,yy,yys)

      real(kind=real8), dimension(:), pointer :: xx, xxs, yy, yys
      character(len=80) :: filename

      real(kind=real8)   :: xi,xsi,yi,ysi    
      integer :: ioresult
      integer, save :: ant_par,i=1

      type(linje), pointer :: start, travers

      allocate(start)
      travers=>start
      nullify(travers%neste_linje)
      do
         write(unit=*,fmt="(a)",advance="no") "Input file >"
         read(unit=*,fmt="(a)") filename
         open(unit=2,status="old",action="read",file=filename,iostat=ioresult)
         if (ioresult/=0) then
            write(unit=*,fmt="(a,a)") "Cannot open file : ", filename
         else
            exit
         endif
      enddo

      do
         read(unit=2,fmt=*,iostat=ioresult) xi, yi,  xsi,  ysi

         if (ioresult < 0) then
            exit
         endif

         travers%x = xi
         travers%xs = xsi
         travers%y = yi
         travers%ys = ysi
         allocate(travers%neste_linje)
         travers=>travers%neste_linje
         ant_par = ant_par + 1
      enddo
      
      close(unit=2)

      nullify(travers%neste_linje)
      travers=>start

      allocate(xx(ant_par))
      allocate(xxs(ant_par))
      allocate(yy(ant_par))
      allocate(yys(ant_par))

      travers=>start
      do
         xx(i)=travers%x
         xxs(i)=travers%xs
         yy(i)=travers%y
         yys(i)=travers%ys
         travers=>travers%neste_linje
         if (.not. associated(travers%neste_linje)) then 
            exit
         endif
         i=i+1
      enddo

      return

    end subroutine lesinn

end module innut




module calculate

  public :: mkm, york, robust, outliers
  private :: rofunc

  contains

    subroutine mkm(a,sigma_a,b,sigma_b,coef)
    
      use data
      real(kind=real8), intent(out) :: a, sigma_a, b, sigma_b, coef      
      real(kind=real8)  :: sum_x, sum_y, sum_xy, sum_x2, sum_y2, sxy, sxx, syy
      real(kind=real8)  :: see, chisq
      integer :: n

      n=size(x)

      chisq = 0.0 
      sum_x = 0.0
      sum_y = 0.0
      sum_xy = 0.0
      sum_x2 = 0.0
      sum_y2 = 0.0
    
      sum_x = sum(x)
      sum_y = sum(y)
      sum_xy = sum(x*y)
      sum_x2 = sum(x*x)
      sum_y2 = sum(y*y)

      sxx = sum_x2 - sum_x * sum_x / n
      sxy = sum_xy - sum_x * sum_y / n
      syy = sum_y2 - sum_y * sum_y / n
      b = sxy / sxx
      a = ((sum_x2*sum_y-sum_x*sum_xy)/n)/sxx

      coef = sxy/sqrt(sxx*syy)
      see = sqrt((sum_y2 - a*sum_y - b*sum_xy)/(n-2))
      sigma_b = see/sqrt(sxx)
      sigma_a = sigma_b * sqrt(sum_x2/n)

      y_calc = a+b*x
      chisq = sum((y-y_calc)**2)

      sigma_a = sigma_a*sqrt(chisq/(n-2))
      sigma_b = sigma_b*sqrt(chisq/(n-2))
      

    end subroutine mkm




    subroutine york(a,sigma_a,b,sigma_b)
    
      use data
      real(kind=real8), intent(out) :: a, sigma_a, b, sigma_b

      real(kind=real8), dimension(size(x)) :: omega_x, omega_y
      real(kind=real8), dimension(size(x)) :: U, V, W
      real(kind=real8)  :: sum_V2, sum_U2, sum_UV
      real(kind=real8)  :: Xbar, Ybar
      real(kind=real8)  :: b3, alpha, beta, gamma, cos_th, th
      real(kind=real8), parameter ::  pi=3.14159265358979323846

      omega_x = 1.0/(xs**2)
      omega_y = 1.0/(ys**2)

      Xbar = sum(x)/size(x)
      Ybar = sum(y)/size(y)


      U = x - Xbar
      V = y - Ybar


      sum_V2 = sum(V*V)
      sum_U2 = sum(U*U)
      sum_UV = sum(U*V)

! Major axis first, need an estimate for b      


      b=(sum_V2 - sum_U2 + sqrt((sum_V2-sum_U2)**2 + 4*sum_UV**2))/(2*sum_UV)

!     b : major axis, min sum av perpendikulære kvadratavvikene.

      print *,"b ('major axis') ",b

!      b = -0.5597425699  
! Fra MKM metode, test.

      do

         W = (omega_x * omega_y)/(b*b*omega_y + omega_x)

         Xbar = sum(W*x)/sum(W)
         Ybar = sum(W*y)/sum(W)

         U = x - Xbar
         V = y - Ybar

      
         alpha = (2*sum((W*W/omega_x)*U*V)) / (3*sum(W*W*U*U/omega_x))

         beta = (sum(W*W*V*V/omega_x)-sum(W*U*U)) / (3*sum(W*W*U*U/omega_x))

         gamma = -sum(W*U*V)/(sum(W*W*U*U/omega_x))

         cos_th = (alpha**3-1.5*alpha*beta+0.5*gamma)/((alpha**2-beta)**1.5)

         th = acos(cos_th)

         b3 = alpha + 2*sqrt(alpha**2 - beta)*cos((1.0/3.0)*(th+4*pi)) 

         print *," York b3 ",b3

         if (abs(b3-b)<0.0001) then
            exit
         endif

         b=b3

      enddo

      a = Ybar - Xbar*b

      sigma_b = sqrt((1.0/(size(x)-2))* (sum(W*(b*U-V)**2)/(sum(W*U*U))))

      sigma_a = sqrt(sigma_b**2 * (sum(W*x*x)/sum(W)))

!      print *,"alpha ",alpha
!      print *,"beta ",beta
!      print *,"gamma ",gamma
!      print *,"cos_th ",cos_th
!      print *,"th ", th
!      print *,"b ",b

      y_calc = a + b*x
      

    end subroutine york




    subroutine rofunc(a,b,res)
     
      use data
      real(kind=real8), intent(out) :: a
      real(kind=real8), intent(in) :: b
      real(kind=real8), intent(out) :: res

      real(kind=real8), dimension(size(x)) :: arr
      real(kind=real8) :: aa, abdev, d, sum1, tmp
      integer :: ndatat, i, j, n1, nmh, nml

      sum1 = 0.0
      ndatat = size(x)
      
      n1 = ndatat + 1
      nml = n1/2
      nmh=n1-nml
      
!      do j=1, ndatat
!         arr(j) = y(j)-b*x(j)
!      enddo

      arr = y - b*x

      do i = 1,ndatat - 1
         do j = i+1,ndatat
            if (arr(i)>arr(j)) then
               tmp = arr(i)
               arr(i) = arr(j)
               arr(j) = tmp
            endif
         enddo
      enddo



      aa = 0.5*(arr(nml)+arr(nmh))
      
!      print *,"aa i robfunc ",aa

      abdev = 0.0

      do j=1, ndatat
         d = y(j)-(b*x(j)+aa)
         abdev = abdev + abs(d)
         if (d>0.0) then
            sum1 = sum1 + x(j)
         else
            sum1 = sum1 - x(j)
         endif
      enddo

      a = aa 
      res = sum1       

    end subroutine rofunc
         
         
      

    subroutine robust(a,sigma_a,b,sigma_b)
    
      use data
      real(kind=real8), intent(out) :: a, sigma_a, b, sigma_b

      real(kind=real8) :: sx, sy, sxy, sxx, syy, coef, del, chisq
      real(kind=real8) :: aa, bb, temp, sigb, b1, b2, f, f1, f2
      integer :: ndatat, j
      

      sx = 0.0
      sy = 0.0
      sxy = 0.0
      sxx = 0.0 
      syy = 0.0
      chisq = 0.0

      ndatat = size(x)
      print *,ndatat," datapunkter lest"

      do j = 1, ndatat
         sx = sx + x(j)
         sy = sy + y(j)
         sxy = sxy + x(j)*y(j)
         sxx = sxx + x(j)*x(j)
         syy = syy + y(j)*y(j)
      enddo
      
      coef = sxy/sqrt(sxx*syy)
      del = ndatat*sxx - sx*sx
      aa = (sxx*sy-sx*sxy)/del
      bb = (ndatat*sxy-sx*sy)/del
         

!      print *,"del :", del
!      print *,"aa  :", aa
!      print *,"bb  :", bb
    
      do j=1,ndatat
         temp = y(j)-(aa+bb*x(j))
         chisq = chisq + temp*temp
      enddo
      sigb = sqrt(chisq/del)
      
      b1 = bb
      call rofunc(aa,b1,f1)
      
!      print *,"f1 :",f1
      
      b2 = bb + sign(3.0*sigb,sigb)
      call rofunc(aa,b2,f2)
      
!      print *,"b2 :", b2
!      print *,"f2 :", f2
      
      do
       if (f1*f2 < 0.0) then
          exit
       endif
       bb = 2.0*b2 - b1
       b1 = b2
       f1 = f2 
       b2 = bb
       call rofunc(aa,b2,f2)
    enddo

    sigb = 0.01*sigb

    do
       if (abs(b2-b1)<sigb) then
          exit
       endif

       bb = 0.5*(b1+b2)
       if (bb .ne. b1 .and. bb .ne. b2) then
          call rofunc(aa,bb,f)
          if (f*f1 .ge. 0.0) then
             f1 = f
             b1 = bb
          else
             f2 = f 
             b2 = bb 
          endif
       endif
    enddo

    a = aa
    b = bb
 
    sigma_a = -1
    sigma_b = -1
    chisq = -1

  end subroutine robust


  
  subroutine outliers(a,b)
    
    use data
    real(kind=real8), intent(in) :: a, b

    real(kind=real8), dimension(:), allocatable :: e, xx, yy
    real(kind=real8) :: tmp
    integer :: i, j, n, kuttes
    
    n=size(x)
    allocate(e(n))
    
    do i=1, n
       e(i) = abs(y(i) - (a+b*x(i)))
    enddo
    do i = 1, n-1
       do j = i+1, n
          if (e(i) > e(j)) then
             tmp = e(i)
             e(i) = e(j)
             e(j) = tmp

             tmp = x(i)
             x(i) = x(j)
             x(j) = tmp

             tmp = y(i)
             y(i) = y(j)
             y(j) = tmp
          endif
       enddo
    enddo
!    print *,"n :", n
    if (n > 100) then
       kuttes = n/6
    else
       kuttes = n/4
    endif

!    print *,"kuttes ",kuttes
    do j = 1, n-kuttes
       x(j) = x(j+kuttes)
       y(j) = y(j+kuttes)
    enddo
    
    n=n-2*kuttes
!    print *," n-2*kuttes", n

    do i = 1, n-1
       do j = i+1, n
          if (x(i) > x(j)) then
             tmp = x(i)
             x(i) = x(j)
             x(j) = tmp

             tmp = y(i)
             y(i) = y(j)
             y(j) = tmp
          endif
       enddo
    enddo

! Her må vi til med kunstgrep fordi hele programmet baserer seg på
! at arrayenens lengde er lik datalengden.

    allocate(xx(n))
    allocate(yy(n))
    
    do i= 1,n
       xx(i) = x(i)
       yy(i) = y(i)
    enddo

    deallocate(x)
    deallocate(y)
    allocate(x(n))
    allocate(y(n))

! Her kan vi bare sette dem like siden de nå har samme lengde.

    x = xx
    y = yy
   
    deallocate(xx)
    deallocate(yy)
    deallocate(e)

  end subroutine outliers
  

end module calculate




program lsq

  use data
  use innut
  use calculate

  real(kind=real8)  :: a, sigma_a, b, sigma_b, coef
  character(len=40) :: frmt
  character(len=1) :: reg
  
  frmt = "(2(1x,a,g12.6))"

  write(unit=*,advance="no",fmt="(1x,a)") "mkm (m), york (y), robust (r), outliers rem. (o) >"
  read(unit=*,fmt="(a)") reg  
  
  if (reg=="m") then
     call lesinn(x,xs,y,ys) 
     allocate(y_calc(size(x)))
     call mkm(a,sigma_a,b,sigma_b,coef)
     print *,"Least sq :"
     write(unit=*,fmt=frmt) "a= ",a," sigma a= ", sigma_a
     write(unit=*,fmt=frmt) "b= ",b," sigma b= ", sigma_b
     write(unit=*,fmt="(1x,a,f5.3)") "Correl coef ",coef
  endif
  
  if (reg=="y") then
     call lesinn(x,xs,y,ys) 
     allocate(y_calc(size(x)))
     call york(a,sigma_a,b,sigma_b)
     print *,"York least sq :"
     write(unit=*,fmt=frmt) "a= ",a," sigma a= ", sigma_a
     write(unit=*,fmt=frmt) "b= ",b," sigma b= ", sigma_b
  endif

  if (reg=="r") then
     call lesinn(x,xs,y,ys) 
     allocate(y_calc(size(x)))

     call robust(a,sigma_a,b,sigma_b)
     print *,"Robust estimering av linja :"
     write(unit=*,fmt=frmt) "a= ",a
     write(unit=*,fmt=frmt) "b= ",b
  endif


  if (reg=="o") then
     call lesinn(x,xs,y,ys) 
     allocate(y_calc(size(x)))
     call robust(a,sigma_a,b,sigma_b)
     print *,"Robust estimering av linja :"
     write(unit=*,fmt=frmt) "a= ",a
     write(unit=*,fmt=frmt) "b= ",b

     call outliers(a,b)
     call robust(a,sigma_a,b,sigma_b)
     print *,"Robust estimering av linja :"
     write(unit=*,fmt=frmt) "a= ",a
     write(unit=*,fmt=frmt) "b= ",b

  endif


end program lsq





