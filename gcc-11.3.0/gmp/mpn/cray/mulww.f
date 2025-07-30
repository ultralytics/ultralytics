c    Helper for mpn_mul_1, mpn_addmul_1, and mpn_submul_1 for Cray PVP.

c    Copyright 1996, 2000 Free Software Foundation, Inc.

c    This file is part of the GNU MP Library.
c  
c    The GNU MP Library is free software; you can redistribute it and/or modify
c    it under the terms of either:
c  
c      * the GNU Lesser General Public License as published by the Free
c        Software Foundation; either version 3 of the License, or (at your
c        option) any later version.
c  
c    or
c  
c      * the GNU General Public License as published by the Free Software
c        Foundation; either version 2 of the License, or (at your option) any
c        later version.
c  
c    or both in parallel, as here.
c  
c    The GNU MP Library is distributed in the hope that it will be useful, but
c    WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
c    or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
c    for more details.
c  
c    You should have received copies of the GNU General Public License and the
c    GNU Lesser General Public License along with the GNU MP Library.  If not,
c    see https://www.gnu.org/licenses/.

c    p1[] = hi(a[]*s); the upper limbs of each product
c    p0[] = low(a[]*s); the corresponding lower limbs
c    n is number of limbs in the vectors

      subroutine gmpn_mulww(p1,p0,a,n,s)
      integer*8 p1(0:*),p0(0:*),a(0:*),s
      integer n

      integer*8 a0,a1,a2,s0,s1,s2,c
      integer*8 ai,t0,t1,t2,t3,t4

      s0 = shiftl(and(s,4194303),24)
      s1 = shiftl(and(shiftr(s,22),4194303),24)
      s2 = shiftl(and(shiftr(s,44),4194303),24)

      do i = 0,n-1
         ai = a(i)
         a0 = shiftl(and(ai,4194303),24)
         a1 = shiftl(and(shiftr(ai,22),4194303),24)
         a2 = shiftl(and(shiftr(ai,44),4194303),24)

         t0 = i24mult(a0,s0)
         t1 = i24mult(a0,s1)+i24mult(a1,s0)
         t2 = i24mult(a0,s2)+i24mult(a1,s1)+i24mult(a2,s0)
         t3 = i24mult(a1,s2)+i24mult(a2,s1)
         t4 = i24mult(a2,s2)

         p0(i)=shiftl(t2,44)+shiftl(t1,22)+t0
         c=shiftr(shiftr(t0,22)+and(t1,4398046511103)+
     $        shiftl(and(t2,1048575),22),42)
         p1(i)=shiftl(t4,24)+shiftl(t3,2)+shiftr(t2,20)+shiftr(t1,42)+c
      end do
      end
