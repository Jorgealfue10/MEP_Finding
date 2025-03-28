module DIATS_H2
integer, parameter :: nmxHHas=51,mq2mHH=5
real*8 riHHas(nmxHHas),alpha_HHas(nmxHHas)
real*8 psumHHas(4,0:nmxHHas)
!------------------------------------------------------------------------------!
contains
!------------------------------------------------------------------------------!
subroutine readcoefDIATS_H2
integer i
    riHHas(1)=    0.74000000d0
    riHHas(2)=    0.80000000d0
    riHHas(3)=    0.86000000d0
    riHHas(4)=    0.92000000d0
    riHHas(5)=    0.98000000d0
    riHHas(6)=    1.04000000d0
    riHHas(7)=    1.10000000d0
    riHHas(8)=    1.16000000d0
    riHHas(9)=    1.22000000d0
    riHHas(10)=   1.28000000d0
    riHHas(11)=   1.34000000d0
    riHHas(12)=   1.40000000d0
    riHHas(13)=   1.46000000d0
    riHHas(14)=   1.52000000d0
    riHHas(15)=   1.58000000d0
    riHHas(16)=   1.64000000d0
    riHHas(17)=   1.70000000d0
    riHHas(18)=   1.76000000d0
    riHHas(19)=   1.82000000d0
    riHHas(20)=   1.88000000d0
    riHHas(21)=   1.94000000d0
    riHHas(22)=   2.00000000d0
    riHHas(23)=   2.10000000d0
    riHHas(24)=   2.20000000d0
    riHHas(25)=   2.30000000d0
    riHHas(26)=   2.40000000d0
    riHHas(27)=   2.50000000d0
    riHHas(28)=   2.60000000d0
    riHHas(29)=   2.70000000d0
    riHHas(30)=   2.80000000d0
    riHHas(31)=   2.90000000d0
    riHHas(32)=   3.00000000d0
    riHHas(33)=   3.20000000d0
    riHHas(34)=   3.40000000d0
    riHHas(35)=   3.60000000d0
    riHHas(36)=   3.70000000d0
    riHHas(37)=   3.80000000d0
    riHHas(38)=   4.00000000d0
    riHHas(39)=   4.20000000d0
    riHHas(40)=   4.40000000d0
    riHHas(41)=   4.60000000d0
    riHHas(42)=   4.80000000d0
    riHHas(43)=   5.00000000d0
    riHHas(44)=   5.50000000d0
    riHHas(45)=   6.00000000d0
    riHHas(46)=   6.50000000d0
    riHHas(47)=   7.00000000d0
    riHHas(48)=   7.50000000d0
    riHHas(49)=   8.00000000d0
    riHHas(50)=   9.00000000d0
    riHHas(51)=   10.0000000d0

    alpha_HHas(1)=           2.3089333850608629d0   
    alpha_HHas(2)=          -1.9299624322993369d0   
    alpha_HHas(3)=          0.71065172057277937d0   
    alpha_HHas(4)=           1.1587315943333567d0   
    alpha_HHas(5)=         -0.69114261955724088d0   
    alpha_HHas(6)=           1.1891659370468877d0   
    alpha_HHas(7)=          0.32324184281076984d0   
    alpha_HHas(8)=          0.67592919011517283d0   
    alpha_HHas(9)=          0.59007174279329877d0   
    alpha_HHas(10)=         0.61519736015352133d0   
    alpha_HHas(11)=         0.55703036062407729d0   
    alpha_HHas(12)=         0.46231350243772906d0   
    alpha_HHas(13)=         0.30174675400650841d0   
    alpha_HHas(14)=          6.9812126336661534d-002
    alpha_HHas(15)=        -0.25051605405888211d0   
    alpha_HHas(16)=        -0.70255076296823948d0   
    alpha_HHas(17)=         -1.2309076693946253d0   
    alpha_HHas(18)=         -2.0150541590894768d0   
    alpha_HHas(19)=         -2.7982156420561592d0   
    alpha_HHas(20)=         -4.2130438916137161d0   
    alpha_HHas(21)=         -4.3679361455847312d0   
    alpha_HHas(22)=         -9.9449181173665746d0   
    alpha_HHas(23)=         -17.600520339473530d0   
    alpha_HHas(24)=         -24.276693193283180d0   
    alpha_HHas(25)=         -33.774723693539492d0   
    alpha_HHas(26)=         -44.995868755269939d0   
    alpha_HHas(27)=         -58.468706456196927d0   
    alpha_HHas(28)=         -75.996029977726039d0   
    alpha_HHas(29)=         -86.380909336193213d0   
    alpha_HHas(30)=         -141.52163921800457d0   
    alpha_HHas(31)=         -5.3411062192734411d0   
    alpha_HHas(32)=         -469.48897634349765d0   
    alpha_HHas(33)=          391.96262044297282d0   
    alpha_HHas(34)=         -2633.9074670018440d0   
    alpha_HHas(35)=          4036.9155155573953d0   
    alpha_HHas(36)=         -3638.7406994048361d0   
    alpha_HHas(37)=          630.14215465166740d0   
    alpha_HHas(38)=         -302.08402347103015d0   
    alpha_HHas(39)=          32.898612804536469d0   
    alpha_HHas(40)=          102.09669757414315d0   
    alpha_HHas(41)=          281.01998308479364d0   
    alpha_HHas(42)=          174.43965415935764d0   
    alpha_HHas(43)=          745.32673337494782d0   
    alpha_HHas(44)=          998.65651651925964d0   
    alpha_HHas(45)=          700.73340882734294d0   
    alpha_HHas(46)=          454.73743170685987d0   
    alpha_HHas(47)=          163.94189298747904d0   
    alpha_HHas(48)=          84.190613672485355d0   
    alpha_HHas(49)=         -396.86817896406501d0   
    alpha_HHas(50)=          369.97763238825769d0   
    alpha_HHas(51)=         -1628.5959548190970d0   
!
    call v2psum(riHHas,nmxHHas,alpha_HHas,psumHHas,mq2mHH)
!
end subroutine
!------------------------------------------------------------------------------!
     subroutine v2psum(xgr,nx,coef,psum,mq2m)
!     Presommations des coefficients RKHS pour le calcul
!     rapide des fonctions d'interpolation a 2 corps.
!
      implicit none
      real(8) xgr, coef, fac1, fac2, fac3, psum
      integer i, iz, nx,mq2m
      dimension xgr(nx),coef(nx), psum(4,0:nx)
!
      fac1 = 4.d0 / ((mq2m+1.d0)*(mq2m+2.d0))
      fac2 = (mq2m+1.d0) / (mq2m+3.d0)
      fac3 = fac1*fac2
      psum = 0.d0
!
      do 10 iz=0,nx
        do 11 i=1,iz
   11   psum(1,iz) = psum(1,iz) + coef(i)
        psum(1,iz) = fac1 * psum(1,iz)
        do 12 i=iz+1,nx
   12   psum(2,iz) = psum(2,iz) + coef(i) / xgr(i)**(mq2m+1)
        psum(2,iz) = fac1 * psum(2,iz)
        do 13 i=1,iz
   13   psum(3,iz) = psum(3,iz) + coef(i)*xgr(i)
        psum(3,iz) = -fac3 * psum(3,iz)
        do 14 i=iz+1,nx
   14   psum(4,iz) = psum(4,iz) + coef(i) / xgr(i)**(mq2m+2)
        psum(4,iz) = -fac3 * psum(4,iz)
   10 continue
!
      return
      end subroutine
!------------------------------------------------------------------------------!
      double precision function v2fast(y,nx,xgr,psum,ider,drv,mq2m)
!     evaluation de la fonction d'interpolation au point y dans le
!     cas de la mÅthode RP-RKHS
      implicit none
      real(8) xgr, coef, y, drv, psum
      integer i, iz, nx, ider,mq2m
      dimension xgr(nx),psum(4,0:nx)
!
      iz = 1
      do while (xgr(iz).le.y.and.iz.le.nx)
        iz = iz + 1
      end do
      iz = iz - 1
      v2fast = psum(2,iz) + y*psum(4,iz)
      if (y.gt.0.d0) v2fast = v2fast + (y*psum(1,iz) + psum(3,iz)) / y**(mq2m+2)
!
!     calcul de la dÅrivÅe
!
      if (ider.eq.1) then
         drv = psum(4,iz)
         if (y.gt.0.d0) drv  = drv - ((mq2m+1)*y*psum(1,iz) + (mq2m+2)*psum(3,iz)) / y**(mq2m+3)
      end if
!
      return
      end function
!------------------------------------------------------------------------------!
    end module DIATS_H2
!------------------------------------------------------------------------------!
