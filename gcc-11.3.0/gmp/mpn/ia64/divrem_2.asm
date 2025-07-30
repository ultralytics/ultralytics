dnl  IA-64 mpn_divrem_2 -- Divide an mpn number by a normalized 2-limb number.

dnl  Copyright 2010, 2013 Free Software Foundation, Inc.

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

C               norm   frac
C itanium 1
C itanium 2     29     29


C TODO
C  * Inline and interleave limb inversion code with loop setup code.
C  * We should use explicit bundling in much of the code, since it typically
C    cuts some cycles with the GNU assembler.


ASM_START()

C HP's assembler requires these declarations for importing mpn_invert_limb
	.global	mpn_invert_limb
	.type	mpn_invert_limb,@function

C INPUT PARAMETERS
C qp   = r32
C fn   = r33
C np   = r34
C nn   = r35
C dp   = r36

define(`f0x1', `f15')

ASM_START()
PROLOGUE(mpn_divrem_2)
	.prologue
ifdef(`HAVE_ABI_32',
`	addp4		r32 = 0, r32		C M I
	addp4		r34 = 0, r34		C M I
	zxt4		r35 = r35		C I
	addp4		r36 = 0, r36		C M I
	nop.m		0
	zxt4		r33 = r33		C I
	;;
')
	.save ar.pfs, r42
	alloc	 r42 = ar.pfs, 5, 9, 1, 0
	shladd	 r34 = r35, 3, r34
	adds	 r14 = 8, r36
	mov	 r43 = r1
	;;
	adds	 r15 = -8, r34
	ld8	 r39 = [r14]
	.save ar.lc, r45
	mov	 r45 = ar.lc
	adds	 r14 = -16, r34
	mov	 r40 = r0
	adds	 r34 = -24, r34
	;;
	ld8	 r38 = [r15]
	.save rp, r41
	mov	 r41 = b0
	.body
	ld8	 r36 = [r36]
	ld8	 r37 = [r14]
	;;
	cmp.gtu	 p6, p7 = r39, r38
  (p6)	br.cond.dptk .L8
	;;
	cmp.leu	 p8, p9 = r36, r37
	cmp.geu	 p6, p7 = r39, r38
	;;
  (p8)	cmp4.ne.and.orcm p6, p7 = 0, r0
  (p7)	br.cond.dptk .L51
.L8:
	add	 r14 = r33, r35		// un + fn
	mov	 r46 = r39		// argument to mpn_invert_limb
	;;
	adds	 r35 = -3, r14
	;;
	cmp.gt	 p12, p0 = r0, r35
  (p12)	br.cond.dpnt L(end)
	br.call.sptk.many b0 = mpn_invert_limb
	;;
	setf.sig f11 = r8		// di (non-final)
	setf.sig f34 = r39		// d1
	setf.sig f33 = r36		// d0
	mov	 r1 = r43
	;;
	mov	 r17 = 1
	setf.sig f9 = r38		// n2
	xma.l	 f6 = f11, f34, f0	// t0 = LO(di * d1)
	;;
	setf.sig f10 = r37		// n1
	setf.sig f15 = r17		// 1
	xma.hu	 f8 = f11, f33, f0	// s0 = HI(di * d0)
	;;
	getf.sig r17 = f6
	getf.sig r16 = f8
	mov	 ar.lc = r35
	;;
	sub	 r18 = r0, r39		// -d1
	add	 r14 = r17, r36
	;;
	setf.sig f14 = r18		// -d1
	cmp.leu	 p8, p9 = r17, r14
	add	 r16 = r14, r16
	;;
  (p9)	adds	 r19 = 0, r0
  (p8)	adds	 r19 = -1, r0
	cmp.gtu	 p6, p7 = r14, r16
	;;
  (p6)	adds	 r19 = 1, r19
	;;
ifelse(1,1,`
	cmp.gt	 p7, p6 = r0, r19
	;;
  (p6)	adds	 r8 = -1, r8		// di--
  (p6)	sub	 r14 = r16, r39		// t0 -= d1
  (p6)	cmp.ltu	 p6, p7 = r16, r39	// cy for: t0 - d1
	;;
  (p6)	cmp.gt	 p9, p8 = 1, r19
  (p7)	cmp.gt	 p9, p8 = 0, r19
  (p6)	adds	 r19 = -1, r19		// t1 -= cy
	mov	 r16 = r14
	;;
  (p8)	adds	 r8 = -1, r8		// di--
  (p8)	sub	 r14 = r16, r39		// t0 -= d1
  (p8)	cmp.ltu	 p8, p9 = r16, r39	// cy for: t0 - d1
	;;
  (p8)	cmp.gt	 p7, p6 = 1, r19
  (p9)	cmp.gt	 p7, p6 = 0, r19
  (p8)	adds	 r19 = -1, r19		// t1 -= cy
	mov	 r16 = r14
	;;
  (p6)	adds	 r8 = -1, r8		// di--
  (p6)	sub	 r14 = r16, r39		// t0 -= d1
  (p6)	cmp.ltu	 p6, p7 = r16, r39	// cy for: t0 - d1
	;;
  (p6)	cmp.gt	 p9, p8 = 1, r19
  (p7)	cmp.gt	 p9, p8 = 0, r19
  (p6)	adds	 r19 = -1, r19		// t1 -= cy
	mov	 r16 = r14
	;;
  (p8)	adds	 r8 = -1, r8		// di--
  (p8)	sub	 r14 = r16, r39		// t0 -= d1
  (p8)	cmp.ltu	 p8, p9 = r16, r39	// cy for: t0 - d1
	;;
  (p8)	adds	 r19 = -1, r19		// t1 -= cy
	mov	 r16 = r14
',`
	cmp.gt	 p8, p9 = r0, r19
  (p8)	br.cond.dpnt .L46
.L52:
	cmp.leu	 p6, p7 = r39, r16
	sub	 r14 = r16, r39
	adds	 r8 = -1, r8
	;;
  (p7)	adds	 r19 = -1, r19
	mov	 r16 = r14
	;;
  (p7)	cmp.gt	 p8, p9 = r0, r19
  (p9)	br.cond.dptk .L52
.L46:
')
	setf.sig f32 = r8		// di
	shladd	 r32 = r35, 3, r32
	;;

	ALIGN(16)
L(top):	nop 0
	nop 0
	cmp.gt	 p8, p9 = r33, r35
	;;
 (p8)	mov	 r37 = r0
 (p9)	ld8	 r37 = [r34], -8
	xma.hu	 f8 = f9, f32, f10	//				0,29
	xma.l	 f12 = f9, f32, f10	//				0
	;;
	getf.sig r20 = f12		// q0				4
	xma.l	 f13 = f15, f8, f9	// q += n2			4
	sub	 r8 = -1, r36		// bitnot d0
	;;
	getf.sig r18 = f13		//				8
	xma.l	 f7 = f14, f13, f10	//				8
	xma.l	 f6 = f33, f13, f33	// t0 = LO(d0*q+d0)		8
	xma.hu	 f9 = f33, f13, f33	// t1 = HI(d0*q+d0)		9
	;;
	getf.sig r38 = f7		// n1				12
	getf.sig r16 = f6		//				13
	getf.sig r19 = f9		//				14
	;;
	sub	 r38 = r38, r39		// n1 -= d1			17
	;;
	cmp.ne	 p9, p0 = r0, r0	// clear p9
	cmp.leu	 p10, p11 = r16, r37	// cy for: n0 - t0		18
	;;
	sub	 r37 = r37, r16		// n0 -= t0			19
  (p11)	sub	 r38 = r38, r19, 1	// n1 -= t1 - cy		19
  (p10)	sub	 r38 = r38, r19		// n1 -= t1			19
	;;
	cmp.gtu	 p6, p7 = r20, r38	// n1 >= q0			20
	;;
  (p7)	cmp.ltu	 p9, p0 = r8, r37	//				21
  (p6)	add	 r18 = 1, r18		//
  (p7)	add	 r37 = r37, r36		//				21
  (p7)	add	 r38 = r38, r39		//				21
	;;
	setf.sig f10 = r37		// n1				22
  (p9)	add	 r38 = 1, r38		//				22
	;;
	setf.sig f9 = r38		// n2				23
	cmp.gtu	 p6, p7 = r39, r38	//				23
  (p7)	br.cond.spnt L(fix)
L(bck):	st8	 [r32] = r18, -8
	adds	 r35 = -1, r35
	br.cloop.sptk.few L(top)
	;;

L(end):	add	r14 = 8, r34
	add	r15 = 16, r34
	mov	 b0 = r41
	;;
	st8	[r14] = r37
	st8	[r15] = r38
	mov	 ar.pfs = r42
	mov	 r8 = r40
	mov	 ar.lc = r45
	br.ret.sptk.many b0
	;;
.L51:
	.pred.rel "mutex", p8, p9
	sub	 r37 = r37, r36
  (p9)	sub	 r38 = r38, r39, 1
  (p8)	sub	 r38 = r38, r39
	adds	 r40 = 1, r0
	br .L8
	;;

L(fix):	cmp.geu	 p6, p7 = r39, r38
	cmp.leu	 p8, p9 = r36, r37
	;;
  (p8)	cmp4.ne.and.orcm p6, p7 = 0, r0
  (p6)	br.cond.dptk L(bck)
	sub	 r37 = r37, r36
  (p9)	sub	 r38 = r38, r39, 1
  (p8)	sub	 r38 = r38, r39
	adds	 r18 = 1, r18
	;;
	setf.sig f9 = r38		// n2
	setf.sig f10 = r37		// n1
	br	 L(bck)

EPILOGUE()
ASM_END()
