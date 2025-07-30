dnl  IA-64 mpn_add_n_sub_n -- mpn parallel addition and subtraction.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2010 Free Software Foundation, Inc.

dnl  This file is part of the GNU MP Library.
dnl
dnl  The GNU MP Library is free software; you can redistribute it and/or modify
dnl  it under the terms of either:
dnl
dnl    * the GNU Lesser General Public License as published by the Free
dnl      Software Foundation; either version 3 of the License, or (at your
dnl      option) any later version.
dnl
dnl  or
dnl
dnl    * the GNU General Public License as published by the Free Software
dnl      Foundation; either version 2 of the License, or (at your option) any
dnl      later version.
dnl
dnl  or both in parallel, as here.
dnl
dnl  The GNU MP Library is distributed in the hope that it will be useful, but
dnl  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
dnl  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
dnl  for more details.
dnl
dnl  You should have received copies of the GNU General Public License and the
dnl  GNU Lesser General Public License along with the GNU MP Library.  If not,
dnl  see https://www.gnu.org/licenses/.

include(`../config.m4')

C           cycles/limb
C Itanium:      ?
C Itanium 2:    2.25

C INPUT PARAMETERS
define(`sp', `r32')
define(`dp', `r33')
define(`up', `r34')
define(`vp', `r35')
define(`n',  `r36')

C Some useful aliases for registers we use
define(`u0',`r16') define(`u1',`r17') define(`u2',`r18') define(`u3',`r19')
define(`v0',`r20') define(`v1',`r21') define(`v2',`r22') define(`v3',`r23')
define(`s0',`r24') define(`s1',`r25') define(`s2',`r26') define(`s3',`r27')
define(`d0',`r28') define(`d1',`r29') define(`d2',`r30') define(`d3',`r31')
define(`up0',`up')
define(`up1',`r14')
define(`vp0',`vp')
define(`vp1',`r15')


ASM_START()
PROLOGUE(mpn_add_n_sub_n)
	.prologue
	.save	ar.lc, r2
	.body
ifdef(`HAVE_ABI_32',`
	addp4	sp = 0, sp		C				M I
	addp4	dp = 0, dp		C				M I
	nop.i	0
	addp4	up = 0, up		C				M I
	addp4	vp = 0, vp		C				M I
	zxt4	n = n			C				I
	;;
')

	and	r9 = 3, n		C				M I
	mov.i	r2 = ar.lc		C				I0
	add	up1 = 8, up0		C				M I
	add	vp1 = 8, vp0		C				M I
	add	r8 = -2, n		C				M I
	add	r10 = 256, up		C				M I
	;;
	shr.u	r8 = r8, 2		C				I0
	cmp.eq	p10, p0 = 0, r9		C				M I
	cmp.eq	p11, p0 = 2, r9		C				M I
	cmp.eq	p12, p0 = 3, r9		C				M I
	add	r11 = 256, vp		C				M I
	;;
	mov.i	ar.lc = r8		C				I0
  (p10)	br	L(b0)			C				B
  (p11)	br	L(b2)			C				B
  (p12)	br	L(b3)			C				B

L(b1):	ld8	u3 = [up0], 8		C				M01
	add	up1 = 8, up1		C				M I
	cmpltu	p14, p15 = 4, n		C				M I
	ld8	v3 = [vp0], 8		C				M01
	add	vp1 = 8, vp1		C				M I
	;;
	add	s3 = u3, v3		C				M I
	sub	d3 = u3, v3		C				M I
	mov	r8 = 0			C				M I
	;;
	cmpltu	p9, p0 = s3, v3		C  carry from add3		M I
	cmpltu	p13, p0 = u3, v3	C borrow from sub3		M I
  (p15)	br	L(cj1)			C				B
	st8	[sp] = s3, 8		C				M23
	st8	[dp] = d3, 8		C				M23
	br	L(c0)			C				B

L(b0):	cmp.ne	p9, p0 = r0, r0		C				M I
	cmp.ne	p13, p0 = r0, r0	C				M I
L(c0):	ld8	u0 = [up0], 16		C				M01
	ld8	u1 = [up1], 16		C				M01
	;;
	ld8	v0 = [vp0], 16		C				M01
	ld8	v1 = [vp1], 16		C				M01
	;;
	ld8	u2 = [up0], 16		C				M01
	ld8	u3 = [up1], 16		C				M01
	;;
	ld8	v2 = [vp0], 16		C				M01
	ld8	v3 = [vp1], 16		C				M01
	;;
	add	s0 = u0, v0		C				M I
	add	s1 = u1, v1		C				M I
	sub	d0 = u0, v0		C				M I
	sub	d1 = u1, v1		C				M I
	;;
	cmpltu	p6, p0 = s0, v0		C  carry from add0		M I
	cmpltu	p7, p0 = s1, v1		C  carry from add1		M I
	cmpltu	p10, p0 = u0, v0	C borrow from sub0		M I
	cmpltu	p11, p0 = u1, v1	C borrow from sub1		M I
	;;
	nop	0			C
	br.cloop.dptk	L(top)		C				B
	br	L(end)			C				B

L(b3):	ld8	u1 = [up0], 8		C				M01
	add	up1 = 8, up1		C				M I
	ld8	v1 = [vp0], 8		C				M01
	;;
	add	vp1 = 8, vp1		C				M I
	add	s1 = u1, v1		C				M I
	sub	d1 = u1, v1		C				M I
	;;
	cmpltu	p7, p0 = s1, v1		C  carry from add1		M I
	cmpltu	p11, p0 = u1, v1	C borrow from sub1		M I
	;;
	st8	[sp] = s1, 8		C				M23
	st8	[dp] = d1, 8		C				M23
	br	L(c2)			C				B

	ALIGN(32)
L(b2):	cmp.ne	p7, p0 = r0, r0		C				M I
	cmp.ne	p11, p0 = r0, r0	C				M I
	nop	0
L(c2):	ld8	u2 = [up0], 16		C				M01
	ld8	u3 = [up1], 16		C				M01
	cmpltu	p14, p0 = 4, n		C				M I
	;;
	ld8	v2 = [vp0], 16		C				M01
	ld8	v3 = [vp1], 16		C				M01
  (p14)	br	L(gt4)			C				B
	;;
	add	s2 = u2, v2		C				M I
	add	s3 = u3, v3		C				M I
	sub	d2 = u2, v2		C				M I
	sub	d3 = u3, v3		C				M I
	;;
	cmpltu	p8, p0 = s2, v2		C  carry from add0		M I
	cmpltu	p9, p0 = s3, v3		C  carry from add3		M I
	cmpltu	p12, p0 = u2, v2	C borrow from sub2		M I
	cmpltu	p13, p0 = u3, v3	C borrow from sub3		M I
	br	L(cj2)			C				B
	;;
L(gt4):	ld8	u0 = [up0], 16		C				M01
	ld8	u1 = [up1], 16		C				M01
	;;
	ld8	v0 = [vp0], 16		C				M01
	ld8	v1 = [vp1], 16		C				M01
	;;
	add	s2 = u2, v2		C				M I
	add	s3 = u3, v3		C				M I
	sub	d2 = u2, v2		C				M I
	sub	d3 = u3, v3		C				M I
	;;
	cmpltu	p8, p0 = s2, v2		C  carry from add0		M I
	cmpltu	p9, p0 = s3, v3		C  carry from add1		M I
	cmpltu	p12, p0 = u2, v2	C borrow from sub0		M I
	cmpltu	p13, p0 = u3, v3	C borrow from sub1		M I
	br.cloop.dptk	L(mid)		C				B

	ALIGN(32)
L(top):
	ld8	u0 = [up0], 16		C				M01
	ld8	u1 = [up1], 16		C				M01
   (p9)	cmpeqor	p6, p0 = -1, s0		C				M I
   (p9)	add	s0 = 1, s0		C				M I
  (p13)	cmpeqor	p10, p0 = 0, d0		C				M I
  (p13)	add	d0 = -1, d0		C				M I
	;;
	ld8	v0 = [vp0], 16		C				M01
	ld8	v1 = [vp1], 16		C				M01
   (p6)	cmpeqor	p7, p0 = -1, s1		C				M I
   (p6)	add	s1 = 1, s1		C				M I
  (p10)	cmpeqor	p11, p0 = 0, d1		C				M I
  (p10)	add	d1 = -1, d1		C				M I
	;;
	st8	[sp] = s0, 8		C				M23
	st8	[dp] = d0, 8		C				M23
	add	s2 = u2, v2		C				M I
	add	s3 = u3, v3		C				M I
	sub	d2 = u2, v2		C				M I
	sub	d3 = u3, v3		C				M I
	;;
	st8	[sp] = s1, 8		C				M23
	st8	[dp] = d1, 8		C				M23
	cmpltu	p8, p0 = s2, v2		C  carry from add2		M I
	cmpltu	p9, p0 = s3, v3		C  carry from add3		M I
	cmpltu	p12, p0 = u2, v2	C borrow from sub2		M I
	cmpltu	p13, p0 = u3, v3	C borrow from sub3		M I
	;;
L(mid):
	ld8	u2 = [up0], 16		C				M01
	ld8	u3 = [up1], 16		C				M01
   (p7)	cmpeqor	p8, p0 = -1, s2		C				M I
   (p7)	add	s2 = 1, s2		C				M I
  (p11)	cmpeqor	p12, p0 = 0, d2		C				M I
  (p11)	add	d2 = -1, d2		C				M I
	;;
	ld8	v2 = [vp0], 16		C				M01
	ld8	v3 = [vp1], 16		C				M01
   (p8)	cmpeqor	p9, p0 = -1, s3		C				M I
   (p8)	add	s3 = 1, s3		C				M I
  (p12)	cmpeqor	p13, p0 = 0, d3		C				M I
  (p12)	add	d3 = -1, d3		C				M I
	;;
	st8	[sp] = s2, 8		C				M23
	st8	[dp] = d2, 8		C				M23
	add	s0 = u0, v0		C				M I
	add	s1 = u1, v1		C				M I
	sub	d0 = u0, v0		C				M I
	sub	d1 = u1, v1		C				M I
	;;
	st8	[sp] = s3, 8		C				M23
	st8	[dp] = d3, 8		C				M23
	cmpltu	p6, p0 = s0, v0		C  carry from add0		M I
	cmpltu	p7, p0 = s1, v1		C  carry from add1		M I
	cmpltu	p10, p0 = u0, v0	C borrow from sub0		M I
	cmpltu	p11, p0 = u1, v1	C borrow from sub1		M I
	;;
	lfetch	[r10], 32		C				M?
	lfetch	[r11], 32		C				M?
	br.cloop.dptk	L(top)		C				B
	;;

L(end):
	nop	0
	nop	0
   (p9)	cmpeqor	p6, p0 = -1, s0		C				M I
   (p9)	add	s0 = 1, s0		C				M I
  (p13)	cmpeqor	p10, p0 = 0, d0		C				M I
  (p13)	add	d0 = -1, d0		C				M I
	;;
	nop	0
	nop	0
   (p6)	cmpeqor	p7, p0 = -1, s1		C				M I
   (p6)	add	s1 = 1, s1		C				M I
  (p10)	cmpeqor	p11, p0 = 0, d1		C				M I
  (p10)	add	d1 = -1, d1		C				M I
	;;
	st8	[sp] = s0, 8		C				M23
	st8	[dp] = d0, 8		C				M23
	add	s2 = u2, v2		C				M I
	add	s3 = u3, v3		C				M I
	sub	d2 = u2, v2		C				M I
	sub	d3 = u3, v3		C				M I
	;;
	st8	[sp] = s1, 8		C				M23
	st8	[dp] = d1, 8		C				M23
	cmpltu	p8, p0 = s2, v2		C  carry from add2		M I
	cmpltu	p9, p0 = s3, v3		C  carry from add3		M I
	cmpltu	p12, p0 = u2, v2	C borrow from sub2		M I
	cmpltu	p13, p0 = u3, v3	C borrow from sub3		M I
	;;
L(cj2):
   (p7)	cmpeqor	p8, p0 = -1, s2		C				M I
   (p7)	add	s2 = 1, s2		C				M I
  (p11)	cmpeqor	p12, p0 = 0, d2		C				M I
  (p11)	add	d2 = -1, d2		C				M I
	mov	r8 = 0			C				M I
	nop	0
	;;
	st8	[sp] = s2, 8		C				M23
	st8	[dp] = d2, 8		C				M23
   (p8)	cmpeqor	p9, p0 = -1, s3		C				M I
   (p8)	add	s3 = 1, s3		C				M I
  (p12)	cmpeqor	p13, p0 = 0, d3		C				M I
  (p12)	add	d3 = -1, d3		C				M I
	;;
L(cj1):
   (p9)	mov	r8 = 2			C				M I
	;;
	mov.i	ar.lc = r2		C				I0
  (p13)	add	r8 = 1, r8		C				M I
	st8	[sp] = s3		C				M23
	st8	[dp] = d3		C				M23
	br.ret.sptk.many b0		C				B
EPILOGUE()
ASM_END()
