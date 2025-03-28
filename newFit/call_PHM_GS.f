!************************************************************************
      subroutine diat12doub1ap(r,ener,der)
      use DIATS_PH
      implicit real * 8 (a-h,o-z)
      real(8)::r,ener,der
CF2PY real(8),intent(out)::ener,der
      call readcoefDIATS_PH
      ener = v2fast(r,nmxPHas,riPHas,psumPHas,1,der,mq2mPH)
      return
      end
!************************************************************************
