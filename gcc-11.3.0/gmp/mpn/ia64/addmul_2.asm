dnl  IA-64 mpn_addmul_2 -- Multiply a n-limb number with a 2-limb number and
dnl  add the result to a (n+1)-limb number.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2004, 2005, 2011 Free Software Foundation, Inc.

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

C         cycles/limb
C Itanium:    3.65
C Itanium 2:  1.625

C TODO
C  * Clean up variable names, and try to decrease the number of distinct
C    registers used.
C  * Clean up feed-in code to not require zeroing several registers.
C  * Make sure we don't depend on uninitialised predicate registers.
C  * Could perhaps save a few cycles by using 1 c/l carry propagation in
C    wind-down code.
C  * Ultimately rewrite.  The problem with this code is that it first uses a
C    loaded u value in one xma pair, then leaves it live over several unrelated
C    xma pairs, before it uses it again.  It should actually be quite possible
C    to just swap some aligned xma pairs around.  But we should then schedule
C    u loads further from the first use.

C INPUT PARAMETERS
define(`rp',`r32')
define(`up',`r33')
define(`n',`r34')
define(`vp',`r35')

define(`srp',`r3')

define(`v0',`f6')
define(`v1',`f7')

define(`s0',`r14')
define(`acc0',`r15')

define(`pr0_0',`r16') define(`pr0_1',`r17')
define(`pr0_2',`r18') define(`pr0_3',`r19')

define(`pr1_0',`r20') define(`pr1_1',`r21')
define(`pr1_2',`r22') define(`pr1_3',`r23')

define(`acc1_0',`r24') define(`acc1_1',`r25')
define(`acc1_2',`r26') define(`acc1_3',`r27')

dnl define(`',`r28')
dnl define(`',`r29')
dnl define(`',`r30')
dnl define(`',`r31')

define(`fp0b_0',`f8') define(`fp0b_1',`f9')
define(`fp0b_2',`f10') define(`fp0b_3',`f11')

define(`fp1a_0',`f12') define(`fp1a_1',`f13')
define(`fp1a_2',`f14') define(`fp1a_3',`f15')

define(`fp1b_0',`f32') define(`fp1b_1',`f33')
define(`fp1b_2',`f34') define(`fp1b_3',`f35')

define(`fp2a_0',`f36') define(`fp2a_1',`f37')
define(`fp2a_2',`f38') define(`fp2a_3',`f39')

define(`r_0',`f40') define(`r_1',`f41')
define(`r_2',`f42') define(`r_3',`f43')

define(`u_0',`f44') define(`u_1',`f45')
define(`u_2',`f46') define(`u_3',`f47')

define(`rx',`f48')
define(`ux',`f49')
define(`ry',`f50')
define(`uy',`f51')

ASM_START()
PROLOGUE(mpn_addmul_2s)
	.prologue
	.save	ar.lc, r2
	.body

ifdef(`HAVE_ABI_32',`
 {.mmi;		addp4	rp = 0, rp		C			M I
		addp4	up = 0, up		C			M I
		addp4	vp = 0, vp		C			M I
}{.mmi;		nop	1
		nop	1
		zxt4	n = n			C			I
	;;
}')

 {.mmi;		ldf8	ux = [up], 8		C			M
		ldf8	v0 = [vp], 8		C			M
		mov	r2 = ar.lc		C			I0
}{.mmi;		ldf8	rx = [rp], 8		C			M
		and	r14 = 3, n		C			M I
		add	n = -2, n		C			M I
	;;
}{.mmi;		ldf8	uy = [up], 8		C			M
		ldf8	v1 = [vp]		C			M
		shr.u	n = n, 2		C			I0
}{.mmi;		ldf8	ry = [rp], -8		C			M
		cmp.eq	p14, p0 = 1, r14	C			M I
		cmp.eq	p11, p0 = 2, r14	C			M I
	;;
}{.mmi;		add	srp = 16, rp		C			M I
		cmp.eq	p15, p0 = 3, r14	C			M I
		mov	ar.lc = n		C			I0
}{.bbb;	(p14)	br.dptk	L(x01)			C			B
	(p11)	br.dptk	L(x10)			C			B
	(p15)	br.dptk	L(x11)			C			B
	;;
}
L(x00):		cmp.ne	p6, p0 = r0, r0		C suppress initial xma pair
		mov	fp2a_3 = f0
		br	L(b00)
L(x01):		cmp.ne	p14, p0 = r0, r0	C suppress initial xma pair
		mov	fp2a_2 = f0
		br	L(b01)
L(x10):		cmp.ne	p11, p0 = r0, r0	C suppress initial xma pair
		mov	fp2a_1 = f0
		br	L(b10)
L(x11):		cmp.ne	p15, p0 = r0, r0	C suppress initial xma pair
		mov	fp2a_0 = f0
		br	L(b11)

EPILOGUE()

PROLOGUE(mpn_addmul_2)
	.prologue
	.save	ar.lc, r2
	.body

ifdef(`HAVE_ABI_32',`
 {.mmi;		addp4	rp = 0, rp		C			M I
		addp4	up = 0, up		C			M I
		addp4	vp = 0, vp		C			M I
}{.mmi;		nop	1
		nop	1
		zxt4	n = n			C			I
	;;
}')

 {.mmi;		ldf8	ux = [up], 8		C			M
		ldf8	v0 = [vp], 8		C			M
		mov	r2 = ar.lc		C			I0
}{.mmi;		ldf8	rx = [rp], 8		C			M
		and	r14 = 3, n		C			M I
		add	n = -2, n		C			M I
	;;
}{.mmi;		ldf8	uy = [up], 8		C			M
		ldf8	v1 = [vp]		C			M
		shr.u	n = n, 2		C			I0
}{.mmi;		ldf8	ry = [rp], -8		C			M
		cmp.eq	p14, p0 = 1, r14	C			M I
		cmp.eq	p11, p0 = 2, r14	C			M I
	;;
}{.mmi;		add	srp = 16, rp		C			M I
		cmp.eq	p15, p6 = 3, r14	C			M I
		mov	ar.lc = n		C			I0
}{.bbb;	(p14)	br.dptk	L(b01)			C			B
	(p11)	br.dptk	L(b10)			C			B
	(p15)	br.dptk	L(b11)			C			B
	;;
}
	ALIGN(32)
L(b00):
 {.mmi;		ldf8	r_1 = [srp], 8
		ldf8	u_1 = [up], 8
		mov	acc1_2 = 0
}{.mmi;		mov	pr1_2 = 0
		mov	pr0_3 = 0
		cmp.ne	p8, p9 = r0, r0
	;;
}{.mfi;		ldf8	r_2 = [srp], 8
		xma.l	fp0b_3 = ux, v0, rx
		cmp.ne	p12, p13 = r0, r0
}{.mfb;		ldf8	u_2 = [up], 8
		xma.hu	fp1b_3 = ux, v0, rx
		br.cloop.dptk	L(gt4)
}
		xma.l	fp0b_0 = uy, v0, ry
		xma.hu	fp1a_0 = uy, v0, ry
	;;
		getfsig	acc0 = fp0b_3
	(p6)	xma.hu	fp2a_3 = ux, v1, fp1b_3		C suppressed for addmul_2s
	(p6)	xma.l	fp1b_3 = ux, v1, fp1b_3		C suppressed for addmul_2s
	;;
		xma.l	fp0b_1 = u_1, v0, r_1
		xma.hu	fp1a_1 = u_1, v0, r_1
	;;
		getfsig	pr0_0 = fp0b_0
		xma.l	fp1b_0 = uy, v1, fp1a_0
		xma.hu	fp2a_0 = uy, v1, fp1a_0
	;;
		getfsig	pr1_3 = fp1b_3
		getfsig	acc1_3 = fp2a_3
		xma.l	fp0b_2 = u_2, v0, r_2
		xma.hu	fp1a_2 = u_2, v0, r_2
		br	L(cj4)

L(gt4):		xma.l	fp0b_0 = uy, v0, ry
		xma.hu	fp1a_0 = uy, v0, ry
	;;
		ldf8	r_3 = [srp], 8
		getfsig	acc0 = fp0b_3
	(p6)	xma.hu	fp2a_3 = ux, v1, fp1b_3		C suppressed for addmul_2s
		ldf8	u_3 = [up], 8
	(p6)	xma.l	fp1b_3 = ux, v1, fp1b_3		C suppressed for addmul_2s
	;;
		xma.l	fp0b_1 = u_1, v0, r_1
		xma.hu	fp1a_1 = u_1, v0, r_1
	;;
		ldf8	r_0 = [srp], 8
		getfsig	pr0_0 = fp0b_0
		xma.l	fp1b_0 = uy, v1, fp1a_0
		xma.hu	fp2a_0 = uy, v1, fp1a_0
	;;
		ldf8	u_0 = [up], 8
		getfsig	pr1_3 = fp1b_3
		xma.l	fp0b_2 = u_2, v0, r_2
	;;
		getfsig	acc1_3 = fp2a_3
		xma.hu	fp1a_2 = u_2, v0, r_2
		br	L(00)


	ALIGN(32)
L(b01):
 {.mmi;		ldf8	r_0 = [srp], 8		C M
		ldf8	u_0 = [up], 8		C M
		mov	acc1_1 = 0		C M I
}{.mmi;		mov	pr1_1 = 0		C M I
		mov	pr0_2 = 0		C M I
		cmp.ne	p6, p7 = r0, r0		C M I
	;;
}{.mfi;		ldf8	r_1 = [srp], 8		C M
		xma.l	fp0b_2 = ux, v0, rx	C F
		cmp.ne	p10, p11 = r0, r0	C M I
}{.mfi;		ldf8	u_1 = [up], 8		C M
		xma.hu	fp1b_2 = ux, v0, rx	C F
		nop	1
	;;
}		xma.l	fp0b_3 = uy, v0, ry	C F
		xma.hu	fp1a_3 = uy, v0, ry	C F
	;;
 {.mmf;		getfsig	acc0 = fp0b_2		C M
		ldf8	r_2 = [srp], 8		C M
	(p14)	xma.hu	fp2a_2 = ux, v1,fp1b_2	C F	suppressed for addmul_2s
}{.mfb;		ldf8	u_2 = [up], 8		C M
	(p14)	xma.l	fp1b_2 = ux, v1,fp1b_2	C F	suppressed for addmul_2s
		br.cloop.dptk	L(gt5)
}
		xma.l	fp0b_0 = u_0, v0, r_0	C F
		xma.hu	fp1a_0 = u_0, v0, r_0	C F
	;;
		getfsig	pr0_3 = fp0b_3		C M
		xma.l	fp1b_3 = uy, v1,fp1a_3	C F
		xma.hu	fp2a_3 = uy, v1,fp1a_3	C F
	;;
		getfsig	pr1_2 = fp1b_2		C M
		getfsig	acc1_2 = fp2a_2		C M
		xma.l	fp0b_1 = u_1, v0, r_1	C F
		xma.hu	fp1a_1 = u_1, v0, r_1	C F
		br	L(cj5)

L(gt5):		xma.l	fp0b_0 = u_0, v0, r_0
		xma.hu	fp1a_0 = u_0, v0, r_0
	;;
		getfsig	pr0_3 = fp0b_3
		ldf8	r_3 = [srp], 8
		xma.l	fp1b_3 = uy, v1, fp1a_3
		xma.hu	fp2a_3 = uy, v1, fp1a_3
	;;
		ldf8	u_3 = [up], 8
		getfsig	pr1_2 = fp1b_2
		xma.l	fp0b_1 = u_1, v0, r_1
	;;
		getfsig	acc1_2 = fp2a_2
		xma.hu	fp1a_1 = u_1, v0, r_1
		br	L(01)


	ALIGN(32)
L(b10):		br.cloop.dptk	L(gt2)
		xma.l	fp0b_1 = ux, v0, rx
		xma.hu	fp1b_1 = ux, v0, rx
	;;
		xma.l	fp0b_2 = uy, v0, ry
		xma.hu	fp1a_2 = uy, v0, ry
	;;
		stf8	[rp] = fp0b_1, 8
	(p11)	xma.hu	fp2a_1 = ux, v1, fp1b_1		C suppressed for addmul_2s
	(p11)	xma.l	fp1b_1 = ux, v1, fp1b_1		C suppressed for addmul_2s
	;;
		getfsig	acc0 = fp0b_2
		xma.l	fp1b_2 = uy, v1, fp1a_2
		xma.hu	fp2a_2 = uy, v1, fp1a_2
	;;
		getfsig	pr1_1 = fp1b_1
		getfsig	acc1_1 = fp2a_1
		mov	ar.lc = r2
		getfsig	pr1_2 = fp1b_2
		getfsig	r8 = fp2a_2
	;;
		add	s0 = pr1_1, acc0
	;;
		st8	[rp] = s0, 8
		cmp.ltu	p8, p9 = s0, pr1_1
		sub	r31 = -1, acc1_1
	;;
	.pred.rel "mutex", p8, p9
	(p8)	add	acc0 = pr1_2, acc1_1, 1
	(p9)	add	acc0 = pr1_2, acc1_1
	(p8)	cmp.leu	p10, p0 = r31, pr1_2
	(p9)	cmp.ltu	p10, p0 = r31, pr1_2
	;;
		st8	[rp] = acc0, 8
	(p10)	add	r8 = 1, r8
		br.ret.sptk.many b0


L(gt2):
 {.mmi;		ldf8	r_3 = [srp], 8
		ldf8	u_3 = [up], 8
		mov	acc1_0 = 0
	;;
}{.mfi;		ldf8	r_0 = [srp], 8
		xma.l	fp0b_1 = ux, v0, rx
		mov	pr1_0 = 0
}{.mfi;		ldf8	u_0 = [up], 8
		xma.hu	fp1b_1 = ux, v0, rx
		mov	pr0_1 = 0
	;;
}		xma.l	fp0b_2 = uy, v0, ry
		xma.hu	fp1a_2 = uy, v0, ry
	;;
		getfsig	acc0 = fp0b_1
		ldf8	r_1 = [srp], 8
	(p11)	xma.hu	fp2a_1 = ux, v1, fp1b_1		C suppressed for addmul_2s
	(p11)	xma.l	fp1b_1 = ux, v1, fp1b_1		C suppressed for addmul_2s
	;;
		ldf8	u_1 = [up], 8
		xma.l	fp0b_3 = u_3, v0, r_3
		xma.hu	fp1a_3 = u_3, v0, r_3
	;;
		getfsig	pr0_2 = fp0b_2
		ldf8	r_2 = [srp], 8
		xma.l	fp1b_2 = uy, v1, fp1a_2
		xma.hu	fp2a_2 = uy, v1, fp1a_2
	;;
		ldf8	u_2 = [up], 8
		getfsig	pr1_1 = fp1b_1
	;;
 {.mfi;		getfsig	acc1_1 = fp2a_1
		xma.l	fp0b_0 = u_0, v0, r_0
		cmp.ne	p8, p9 = r0, r0
}{.mfb;		cmp.ne	p12, p13 = r0, r0
		xma.hu	fp1a_0 = u_0, v0, r_0
		br.cloop.sptk.clr	L(top)
}
		br.many	L(end)


	ALIGN(32)
L(b11):		ldf8	r_2 = [srp], 8
		mov	pr1_3 = 0
		mov	pr0_0 = 0
	;;
		ldf8	u_2 = [up], 8
		mov	acc1_3 = 0
		br.cloop.dptk	L(gt3)
	;;
		cmp.ne	p6, p7 = r0, r0
		xma.l	fp0b_0 = ux, v0, rx
		xma.hu	fp1b_0 = ux, v0, rx
	;;
		cmp.ne	p10, p11 = r0, r0
		xma.l	fp0b_1 = uy, v0, ry
		xma.hu	fp1a_1 = uy, v0, ry
	;;
		getfsig	acc0 = fp0b_0
	(p15)	xma.hu	fp2a_0 = ux, v1, fp1b_0		C suppressed for addmul_2s
	(p15)	xma.l	fp1b_0 = ux, v1, fp1b_0		C suppressed for addmul_2s
	;;
		xma.l	fp0b_2 = uy, v1, r_2
		xma.hu	fp1a_2 = uy, v1, r_2
	;;
		getfsig	pr0_1 = fp0b_1
		xma.l	fp1b_1 = u_2, v0, fp1a_1
		xma.hu	fp2a_1 = u_2, v0, fp1a_1
	;;
		getfsig	pr1_0 = fp1b_0
		getfsig	acc1_0 = fp2a_0
		br	L(cj3)

L(gt3):		ldf8	r_3 = [srp], 8
		xma.l	fp0b_0 = ux, v0, rx
		cmp.ne	p10, p11 = r0, r0
		ldf8	u_3 = [up], 8
		xma.hu	fp1b_0 = ux, v0, rx
		cmp.ne	p6, p7 = r0, r0
	;;
		xma.l	fp0b_1 = uy, v0, ry
		xma.hu	fp1a_1 = uy, v0, ry
	;;
		getfsig	acc0 = fp0b_0
		ldf8	r_0 = [srp], 8
	(p15)	xma.hu	fp2a_0 = ux, v1, fp1b_0		C suppressed for addmul_2s
		ldf8	u_0 = [up], 8
	(p15)	xma.l	fp1b_0 = ux, v1, fp1b_0		C suppressed for addmul_2s
	;;
		xma.l	fp0b_2 = u_2, v0, r_2
		xma.hu	fp1a_2 = u_2, v0, r_2
	;;
		getfsig	pr0_1 = fp0b_1
		ldf8	r_1 = [srp], 8
		xma.l	fp1b_1 = uy, v1, fp1a_1
		xma.hu	fp2a_1 = uy, v1, fp1a_1
	;;
		ldf8	u_1 = [up], 8
		getfsig	pr1_0 = fp1b_0
	;;
		getfsig	acc1_0 = fp2a_0
		xma.l	fp0b_3 = u_3, v0, r_3
		xma.hu	fp1a_3 = u_3, v0, r_3
		br	L(11)


C *** MAIN LOOP START ***
	ALIGN(32)
L(top):						C 00
	.pred.rel "mutex", p12, p13
		getfsig	pr0_3 = fp0b_3
		ldf8	r_3 = [srp], 8
		xma.l	fp1b_3 = u_3, v1, fp1a_3
	(p12)	add	s0 = pr1_0, acc0, 1
	(p13)	add	s0 = pr1_0, acc0
		xma.hu	fp2a_3 = u_3, v1, fp1a_3
	;;					C 01
	.pred.rel "mutex", p8, p9
	.pred.rel "mutex", p12, p13
		ldf8	u_3 = [up], 8
		getfsig	pr1_2 = fp1b_2
	(p8)	cmp.leu	p6, p7 = acc0, pr0_1
	(p9)	cmp.ltu	p6, p7 = acc0, pr0_1
	(p12)	cmp.leu	p10, p11 = s0, pr1_0
	(p13)	cmp.ltu	p10, p11 = s0, pr1_0
	;;					C 02
	.pred.rel "mutex", p6, p7
		getfsig	acc1_2 = fp2a_2
		st8	[rp] = s0, 8
		xma.l	fp0b_1 = u_1, v0, r_1
	(p6)	add	acc0 = pr0_2, acc1_0, 1
	(p7)	add	acc0 = pr0_2, acc1_0
		xma.hu	fp1a_1 = u_1, v0, r_1
	;;					C 03
L(01):
	.pred.rel "mutex", p10, p11
		getfsig	pr0_0 = fp0b_0
		ldf8	r_0 = [srp], 8
		xma.l	fp1b_0 = u_0, v1, fp1a_0
	(p10)	add	s0 = pr1_1, acc0, 1
	(p11)	add	s0 = pr1_1, acc0
		xma.hu	fp2a_0 = u_0, v1, fp1a_0
	;;					C 04
	.pred.rel "mutex", p6, p7
	.pred.rel "mutex", p10, p11
		ldf8	u_0 = [up], 8
		getfsig	pr1_3 = fp1b_3
	(p6)	cmp.leu	p8, p9 = acc0, pr0_2
	(p7)	cmp.ltu	p8, p9 = acc0, pr0_2
	(p10)	cmp.leu	p12, p13 = s0, pr1_1
	(p11)	cmp.ltu	p12, p13 = s0, pr1_1
	;;					C 05
	.pred.rel "mutex", p8, p9
		getfsig	acc1_3 = fp2a_3
		st8	[rp] = s0, 8
		xma.l	fp0b_2 = u_2, v0, r_2
	(p8)	add	acc0 = pr0_3, acc1_1, 1
	(p9)	add	acc0 = pr0_3, acc1_1
		xma.hu	fp1a_2 = u_2, v0, r_2
	;;					C 06
L(00):
	.pred.rel "mutex", p12, p13
		getfsig	pr0_1 = fp0b_1
		ldf8	r_1 = [srp], 8
		xma.l	fp1b_1 = u_1, v1, fp1a_1
	(p12)	add	s0 = pr1_2, acc0, 1
	(p13)	add	s0 = pr1_2, acc0
		xma.hu	fp2a_1 = u_1, v1, fp1a_1
	;;					C 07
	.pred.rel "mutex", p8, p9
	.pred.rel "mutex", p12, p13
		ldf8	u_1 = [up], 8
		getfsig	pr1_0 = fp1b_0
	(p8)	cmp.leu	p6, p7 = acc0, pr0_3
	(p9)	cmp.ltu	p6, p7 = acc0, pr0_3
	(p12)	cmp.leu	p10, p11 = s0, pr1_2
	(p13)	cmp.ltu	p10, p11 = s0, pr1_2
	;;					C 08
	.pred.rel "mutex", p6, p7
		getfsig	acc1_0 = fp2a_0
		st8	[rp] = s0, 8
		xma.l	fp0b_3 = u_3, v0, r_3
	(p6)	add	acc0 = pr0_0, acc1_2, 1
	(p7)	add	acc0 = pr0_0, acc1_2
		xma.hu	fp1a_3 = u_3, v0, r_3
	;;					C 09
L(11):
	.pred.rel "mutex", p10, p11
		getfsig	pr0_2 = fp0b_2
		ldf8	r_2 = [srp], 8
		xma.l	fp1b_2 = u_2, v1, fp1a_2
	(p10)	add	s0 = pr1_3, acc0, 1
	(p11)	add	s0 = pr1_3, acc0
		xma.hu	fp2a_2 = u_2, v1, fp1a_2
	;;					C 10
	.pred.rel "mutex", p6, p7
	.pred.rel "mutex", p10, p11
		ldf8	u_2 = [up], 8
		getfsig	pr1_1 = fp1b_1
	(p6)	cmp.leu	p8, p9 = acc0, pr0_0
	(p7)	cmp.ltu	p8, p9 = acc0, pr0_0
	(p10)	cmp.leu	p12, p13 = s0, pr1_3
	(p11)	cmp.ltu	p12, p13 = s0, pr1_3
	;;					C 11
	.pred.rel "mutex", p8, p9
		getfsig	acc1_1 = fp2a_1
		st8	[rp] = s0, 8
		xma.l	fp0b_0 = u_0, v0, r_0
	(p8)	add	acc0 = pr0_1, acc1_3, 1
	(p9)	add	acc0 = pr0_1, acc1_3
		xma.hu	fp1a_0 = u_0, v0, r_0
L(10):		br.cloop.sptk.clr	L(top)	C 12
	;;
C *** MAIN LOOP END ***
L(end):
	.pred.rel "mutex", p12, p13
 {.mfi;		getfsig	pr0_3 = fp0b_3
		xma.l	fp1b_3 = u_3, v1, fp1a_3
	(p12)	add	s0 = pr1_0, acc0, 1
}{.mfi;	(p13)	add	s0 = pr1_0, acc0
		xma.hu	fp2a_3 = u_3, v1, fp1a_3
		nop	1
	;;
}	.pred.rel "mutex", p8, p9
	.pred.rel "mutex", p12, p13
 {.mmi;		getfsig	pr1_2 = fp1b_2
		st8	[rp] = s0, 8
	(p8)	cmp.leu	p6, p7 = acc0, pr0_1
}{.mmi;	(p9)	cmp.ltu	p6, p7 = acc0, pr0_1
	(p12)	cmp.leu	p10, p11 = s0, pr1_0
	(p13)	cmp.ltu	p10, p11 = s0, pr1_0
	;;
}	.pred.rel "mutex", p6, p7
 {.mfi;		getfsig	acc1_2 = fp2a_2
		xma.l	fp0b_1 = u_1, v0, r_1
		nop	1
}{.mmf;	(p6)	add	acc0 = pr0_2, acc1_0, 1
	(p7)	add	acc0 = pr0_2, acc1_0
		xma.hu	fp1a_1 = u_1, v0, r_1
	;;
}
L(cj5):
	.pred.rel "mutex", p10, p11
 {.mfi;		getfsig	pr0_0 = fp0b_0
		xma.l	fp1b_0 = u_0, v1, fp1a_0
	(p10)	add	s0 = pr1_1, acc0, 1
}{.mfi;	(p11)	add	s0 = pr1_1, acc0
		xma.hu	fp2a_0 = u_0, v1, fp1a_0
		nop	1
	;;
}	.pred.rel "mutex", p6, p7
	.pred.rel "mutex", p10, p11
 {.mmi;		getfsig	pr1_3 = fp1b_3
	st8	[rp] = s0, 8
	(p6)	cmp.leu	p8, p9 = acc0, pr0_2
}{.mmi;	(p7)	cmp.ltu	p8, p9 = acc0, pr0_2
	(p10)	cmp.leu	p12, p13 = s0, pr1_1
	(p11)	cmp.ltu	p12, p13 = s0, pr1_1
	;;
}	.pred.rel "mutex", p8, p9
 {.mfi;		getfsig	acc1_3 = fp2a_3
		xma.l	fp0b_2 = u_2, v0, r_2
		nop	1
}{.mmf;	(p8)	add	acc0 = pr0_3, acc1_1, 1
	(p9)	add	acc0 = pr0_3, acc1_1
		xma.hu	fp1a_2 = u_2, v0, r_2
	;;
}
L(cj4):
	.pred.rel "mutex", p12, p13
 {.mfi;		getfsig	pr0_1 = fp0b_1
		xma.l	fp1b_1 = u_1, v1, fp1a_1
	(p12)	add	s0 = pr1_2, acc0, 1
}{.mfi;	(p13)	add	s0 = pr1_2, acc0
		xma.hu	fp2a_1 = u_1, v1, fp1a_1
		nop	1
	;;
}	.pred.rel "mutex", p8, p9
	.pred.rel "mutex", p12, p13
 {.mmi;		getfsig	pr1_0 = fp1b_0
		st8	[rp] = s0, 8
	(p8)	cmp.leu	p6, p7 = acc0, pr0_3
}{.mmi;	(p9)	cmp.ltu	p6, p7 = acc0, pr0_3
	(p12)	cmp.leu	p10, p11 = s0, pr1_2
	(p13)	cmp.ltu	p10, p11 = s0, pr1_2
	;;
}	.pred.rel "mutex", p6, p7
 {.mmi;		getfsig	acc1_0 = fp2a_0
	(p6)	add	acc0 = pr0_0, acc1_2, 1
	(p7)	add	acc0 = pr0_0, acc1_2
	;;
}
L(cj3):
	.pred.rel "mutex", p10, p11
 {.mfi;		getfsig	pr0_2 = fp0b_2
		xma.l	fp1b_2 = u_2, v1, fp1a_2
	(p10)	add	s0 = pr1_3, acc0, 1
}{.mfi;	(p11)	add	s0 = pr1_3, acc0
		xma.hu	fp2a_2 = u_2, v1, fp1a_2
		nop	1
	;;
}	.pred.rel "mutex", p6, p7
	.pred.rel "mutex", p10, p11
 {.mmi;		getfsig	pr1_1 = fp1b_1
		st8	[rp] = s0, 8
	(p6)	cmp.leu	p8, p9 = acc0, pr0_0
}{.mmi;	(p7)	cmp.ltu	p8, p9 = acc0, pr0_0
	(p10)	cmp.leu	p12, p13 = s0, pr1_3
	(p11)	cmp.ltu	p12, p13 = s0, pr1_3
	;;
}	.pred.rel "mutex", p8, p9
 {.mmi;		getfsig	acc1_1 = fp2a_1
	(p8)	add	acc0 = pr0_1, acc1_3, 1
	(p9)	add	acc0 = pr0_1, acc1_3
	;;
}	.pred.rel "mutex", p12, p13
 {.mmi;	(p12)	add	s0 = pr1_0, acc0, 1
	(p13)	add	s0 = pr1_0, acc0
		nop	1
	;;
}	.pred.rel "mutex", p8, p9
	.pred.rel "mutex", p12, p13
 {.mmi;		getfsig	pr1_2 = fp1b_2
		st8	[rp] = s0, 8
	(p8)	cmp.leu	p6, p7 = acc0, pr0_1
}{.mmi;	(p9)	cmp.ltu	p6, p7 = acc0, pr0_1
	(p12)	cmp.leu	p10, p11 = s0, pr1_0
	(p13)	cmp.ltu	p10, p11 = s0, pr1_0
	;;
}	.pred.rel "mutex", p6, p7
 {.mmi;		getfsig	r8 = fp2a_2
	(p6)	add	acc0 = pr0_2, acc1_0, 1
	(p7)	add	acc0 = pr0_2, acc1_0
	;;
}	.pred.rel "mutex", p10, p11
 {.mmi;	(p10)	add	s0 = pr1_1, acc0, 1
	(p11)	add	s0 = pr1_1, acc0
	(p6)	cmp.leu	p8, p9 = acc0, pr0_2
	;;
}	.pred.rel "mutex", p10, p11
 {.mmi;	(p7)	cmp.ltu	p8, p9 = acc0, pr0_2
	(p10)	cmp.leu	p12, p13 = s0, pr1_1
	(p11)	cmp.ltu	p12, p13 = s0, pr1_1
	;;
}	.pred.rel "mutex", p8, p9
 {.mmi;		st8	[rp] = s0, 8
	(p8)	add	acc0 = pr1_2, acc1_1, 1
	(p9)	add	acc0 = pr1_2, acc1_1
	;;
}	.pred.rel "mutex", p8, p9
 {.mmi;	(p8)	cmp.leu	p10, p11 = acc0, pr1_2
	(p9)	cmp.ltu	p10, p11 = acc0, pr1_2
	(p12)	add	acc0 = 1, acc0
	;;
}{.mmi;		st8	[rp] = acc0, 8
	(p12)	cmpeqor	p10, p0 = 0, acc0
		nop	1
	;;
}{.mib;	(p10)	add	r8 = 1, r8
		mov	ar.lc = r2
		br.ret.sptk.many b0
}
EPILOGUE()
ASM_END()
