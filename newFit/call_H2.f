!************************************************************************
      subroutine diatHH(r,ener,der)
      use DIATS_H2
      implicit real * 8 (a-h,o-z)
      real(8)::r,ener,der
CF2PY real(8),intent(out)::ener,der
      call readcoefDIATS_H2
      ener = v2fast(r,nmxHHas,riHHas,psumHHas,1,der,mq2mHH)
      return
      end
!************************************************************************
