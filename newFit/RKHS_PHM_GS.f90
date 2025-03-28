module DIATS_PH
integer, parameter :: nmxPHas=79,mq2mPH=3
real*8 riPHas(nmxPHas),alpha_PHas(nmxPHas)
real*8 psumPHas(4,0:nmxPHas)
!------------------------------------------------------------------------------!
contains
!------------------------------------------------------------------------------!
subroutine readcoefDIATS_PH
integer i
    riPHas(1)=    1.20000000d0
    riPHas(2)=    1.40000000d0
    riPHas(3)=    1.50000000d0
    riPHas(4)=    1.60000000d0
    riPHas(5)=    1.70000000d0
    riPHas(6)=    1.80000000d0
    riPHas(7)=    1.90000000d0
    riPHas(8)=    2.00000000d0
    riPHas(9)=    2.10000000d0
    riPHas(10)=   2.20000000d0
    riPHas(11)=   2.30000000d0
    riPHas(12)=   2.40000000d0
    riPHas(13)=   2.50000000d0
    riPHas(14)=   2.60000000d0
    riPHas(15)=   2.70000000d0
    riPHas(16)=   2.80000000d0
    riPHas(17)=   2.90000000d0
    riPHas(18)=   3.00000000d0
    riPHas(19)=   3.10000000d0
    riPHas(20)=   3.20000000d0
    riPHas(21)=   3.40000000d0
    riPHas(22)=   3.50000000d0
    riPHas(23)=   3.60000000d0
    riPHas(24)=   3.70000000d0
    riPHas(25)=   3.80000000d0
    riPHas(26)=   3.90000000d0
    riPHas(27)=   4.00000000d0
    riPHas(28)=   4.10000000d0
    riPHas(29)=   4.20000000d0
    riPHas(30)=   4.30000000d0
    riPHas(31)=   4.40000000d0
    riPHas(32)=   4.50000000d0
    riPHas(33)=   4.60000000d0
    riPHas(34)=   4.70000000d0
    riPHas(35)=   4.80000000d0
    riPHas(36)=   4.90000000d0
    riPHas(37)=   5.00000000d0
    riPHas(38)=   5.10000000d0
    riPHas(39)=   5.20000000d0
    riPHas(40)=   5.30000000d0
    riPHas(41)=   5.40000000d0
    riPHas(42)=   5.50000000d0
    riPHas(43)=   5.60000000d0
    riPHas(44)=   5.70000000d0
    riPHas(45)=   5.80000000d0
    riPHas(46)=   5.90000000d0
    riPHas(47)=   6.00000000d0
    riPHas(48)=   6.10000000d0
    riPHas(49)=   6.20000000d0
    riPHas(50)=   6.30000000d0
    riPHas(51)=   6.40000000d0
    riPHas(52)=   7.00000000d0
    riPHas(53)=   7.10000000d0
    riPHas(54)=   7.20000000d0
    riPHas(55)=   7.30000000d0
    riPHas(56)=   7.40000000d0
    riPHas(57)=   7.50000000d0
    riPHas(58)=   7.60000000d0
    riPHas(59)=   7.70000000d0
    riPHas(60)=   7.80000000d0
    riPHas(61)=   7.90000000d0
    riPHas(62)=   8.00000000d0
    riPHas(63)=   8.25000000d0
    riPHas(64)=   8.50000000d0
    riPHas(65)=   8.75000000d0
    riPHas(66)=   9.00000000d0
    riPHas(67)=   9.50000000d0
    riPHas(68)=   10.0000000d0
    riPHas(69)=   10.5000000d0
    riPHas(70)=   11.0000000d0
    riPHas(71)=   11.5000000d0
    riPHas(72)=   12.0000000d0
    riPHas(73)=   12.5000000d0
    riPHas(74)=   13.0000000d0
    riPHas(75)=   13.5000000d0
    riPHas(76)=   14.0000000d0
    riPHas(77)=   14.5000000d0
    riPHas(78)=   15.0000000d0
    riPHas(79)=   16.0000000d0


    alpha_PHas(1)=           83.447722253772980d0     
    alpha_PHas(2)=          -95.950190067414525d0     
    alpha_PHas(3)=           66.561901496829066d0     
    alpha_PHas(4)=          -20.242962378133377d0     
    alpha_PHas(5)=           11.196545841024619d0     
    alpha_PHas(6)=          -10.752254250027418d0     
    alpha_PHas(7)=           11.945101857787497d0     
    alpha_PHas(8)=          -9.0897004593525423d0     
    alpha_PHas(9)=         -0.36155378300812063d0     
    alpha_PHas(10)=         -4.5465019958560342d0     
    alpha_PHas(11)=         -4.6767674212715367d0     
    alpha_PHas(12)=         -6.1783528881754517d0     
    alpha_PHas(13)=         -7.3920746351109390d0     
    alpha_PHas(14)=         -8.8568695662664343d0     
    alpha_PHas(15)=         -10.367491234374617d0     
    alpha_PHas(16)=         -12.166043761456393d0     
    alpha_PHas(17)=         -13.479531343081931d0     
    alpha_PHas(18)=         -17.106540227561464d0     
    alpha_PHas(19)=         -12.195136167792395d0     
    alpha_PHas(20)=         -34.713109902688174d0     
    alpha_PHas(21)=         -42.832555261084273d0     
    alpha_PHas(22)=         -17.918733581597234d0     
    alpha_PHas(23)=         -31.692087307792903d0     
    alpha_PHas(24)=         -30.046905413739303d0     
    alpha_PHas(25)=         -32.858255907802238d0     
    alpha_PHas(26)=         -34.317035623872755d0     
    alpha_PHas(27)=         -34.989994042079459d0     
    alpha_PHas(28)=         -35.399306444009461d0     
    alpha_PHas(29)=         -35.040345188496495d0     
    alpha_PHas(30)=         -32.093151672549162d0     
    alpha_PHas(31)=         -29.393169694003053d0     
    alpha_PHas(32)=         -22.750831750688995d0     
    alpha_PHas(33)=         -14.938907696198839d0     
    alpha_PHas(34)=         -3.9175334560276203d0     
    alpha_PHas(35)=          8.5659651296854680d0     
    alpha_PHas(36)=          22.661044957369626d0     
    alpha_PHas(37)=          37.867054757376970d0     
    alpha_PHas(38)=          50.153482661690809d0     
    alpha_PHas(39)=          60.728905513976194d0     
    alpha_PHas(40)=          67.661003414544425d0     
    alpha_PHas(41)=          70.475428234777112d0     
    alpha_PHas(42)=          68.779484023461762d0     
    alpha_PHas(43)=          63.271625080224496d0     
    alpha_PHas(44)=          58.048971540226617d0     
    alpha_PHas(45)=          44.370906104893002d0     
    alpha_PHas(46)=          49.008509628572185d0     
    alpha_PHas(47)=         0.59020945780969347d0     
    alpha_PHas(48)=          127.73117430001577d0     
    alpha_PHas(49)=         -375.11080638682506d0     
    alpha_PHas(50)=          1560.4763491536376d0     
    alpha_PHas(51)=         -1607.1075207690301d0     
    alpha_PHas(52)=          2029.0177265890045d0     
    alpha_PHas(53)=         -2134.3708004514651d0     
    alpha_PHas(54)=          618.02949517525508d0     
    alpha_PHas(55)=         -200.11689792191649d0     
    alpha_PHas(56)=          51.077263864250000d0     
    alpha_PHas(57)=         -8.0171847177825519d0     
    alpha_PHas(58)=         -23.510691517335204d0     
    alpha_PHas(59)=          10.116561042155721d0     
    alpha_PHas(60)=         -15.139114296748307d0     
    alpha_PHas(61)=         -6.1463187628764748d0     
    alpha_PHas(62)=         -7.8572615194143252d0     
    alpha_PHas(63)=         -35.123095094454101d0     
    alpha_PHas(64)=          60.070888772073083d0     
    alpha_PHas(65)=         -315.31521784573511d0     
    alpha_PHas(66)=          646.25504664327730d0     
    alpha_PHas(67)=         -1323.8767096833722d0     
    alpha_PHas(68)=          1709.6847233046728d0     
    alpha_PHas(69)=         -1169.5826653038887d0     
    alpha_PHas(70)=          372.20567011773176d0     
    alpha_PHas(71)=         -105.43858238295383d0     
    alpha_PHas(72)=          29.866700457804139d0     
    alpha_PHas(73)=         -1.3942687261408651d0     
    alpha_PHas(74)=         -6.9459433140057678d0     
    alpha_PHas(75)=          1.7687960872767139d0     
    alpha_PHas(76)=          4.1520511951377648d0     
    alpha_PHas(77)=         -5.1812116361468270d0     
    alpha_PHas(78)=        -0.67247720150585499d0     
    alpha_PHas(79)=          1.4651115397348988d0     
!
    call v2psum(riPHas,nmxPHas,alpha_PHas,psumPHas,mq2mPH)
!
end subroutine
!------------------------------------------------------------------------------!
     subroutine v2psum(xgr,nx,coef,psum,mq2m)
!     Presommations des coefficients RKHS pour le calcul
!     rapide des fonctions d'interpolation a 2 corps.
!
      implicit none
      real(8) xgr, coef, fac1, fac2, fac3
      real(8),intent(out) :: psum
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
    end module DIATS_PH
!------------------------------------------------------------------------------!
