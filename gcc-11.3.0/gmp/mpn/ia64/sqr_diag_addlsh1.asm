dnl  IA-64 mpn_sqr_diag_addlsh1

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2010, 2011 Free Software Foundation, Inc.

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
C Itanium 2:    2	Unrolling could bring it to 1.5 + epsilon

C Exact performance table.  The 2nd line is this code, the 3rd line is ctop-
C less code.  In an assembly sqr_basecase, the ctop-full numbers will become a
C few cycles better since we can mitigate the many I0 instructions.
C
C 1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
C -  20  22  24  26  28  30  32  34  36  38  40  42  44  46  48  50  52  54  56 Needs updating
C -  13  16  17  18  20  21  23  25  26  30  31  31  33  34  36  38  39  42  43

C We should keep in mind that this code takes linear time in a O(n^2) context
C and that it will only be used under SQR_TOOM2_THRESHOLD, which might become
C around 60.  Keeping overhead down for smallish operands (< 10) is more
C important than optimal cycle counts.

C TODO
C  * Make sure we don't depend on uninitialised r-registers, f-registers, or
C  * p-registers.
C  * Optimise by doing first two loop iterations in function header.

C INPUT PARAMETERS
define(`rp_param', `r32')  define(`rp', `r14')		C size: 2n
define(`tp_param', `r33')  define(`tp', `r15')		C size: 2n - 2
define(`up_param', `r34')  define(`up', `r31')		C size: n
define(`n',  `r35')

ifdef(`HAVE_ABI_32',`
	define(`ABI64', `')
	define(`ABI32', `$1')
',`
	define(`ABI64', `$1')
	define(`ABI32', `')
')

ASM_START()
PROLOGUE(mpn_sqr_diag_addlsh1)

	.prologue
	.save	ar.pfs, r2
	.save	ar.lc, r3
	.body

 {.mii;		alloc	r2 = ar.pfs, 4,24,0,24	C			M
		mov	r3 = ar.lc		C			I0
	ABI64(`	nop	4711		')
	ABI32(`	zxt4	n = n		')
}{.mmi;	ABI64(`	mov	tp = tp_param	')	C			M I
	ABI32(`	addp4	tp = 0, tp_param')	C			M I
	ABI64(`	mov	up = up_param	')	C			M I
	ABI32(`	addp4	up = 0, up_param')	C			M I
	ABI64(`	mov	rp = rp_param	')	C			M I
	ABI32(`	addp4	rp = 0, rp_param')	C			M I
	;;
}{.mmi;		ld8	r36 = [tp], 8		C			M
		add	r20 = -2, n		C			M I
		mov	r9 = ar.ec		C			I0
	;;
}{.mmi;		ld8	r32 = [tp], 8		C			M
		mov	r16 = 0			C			M I
		mov	ar.ec = 7		C			I0
	;;
}{.mmi;		nop	4711
		mov	r44 = 0			C			M I
		mov	ar.lc = r20		C			I0
	;;
}{.mii;		mov	r33 = 0
		mov	r10 = pr		C			I0
		mov	pr.rot = 0x30000	C			I0
	;;
}		br.cexit.spnt.few.clr	L(end)

dnl *** MAIN LOOP START ***
	ALIGN(32)
L(top):
 {.mfi;	(p18)	ldf8	f33 = [up], 8		C			M
	(p20)	xma.l	f36 = f35, f35, f42	C			F
	(p41)	cmpequc	p50, p0 = -1, r44	C			M I
}{.mfi;		setfsig	f40 = r16		C			M23
	(p20)	xma.hu	f38 = f35, f35, f42	C			F
	(p23)	add	r50 = r41, r49		C			M I
	;;
}{.mmi;	(p16)	ld8	r36 = [tp], 8		C			M
	(p23)	cmpltu	p40, p0 = r50, r41	C cyout hi		M I
	(p19)	shrp	r45 = r38, r35, 63	C non-critical		I0
}{.mmi;	(p21)	getfsig	r39 = f39		C hi			M2
	(p24)	st8	[rp] = r51, 8		C hi			M23
	(p41)	add	r44 = 1, r44		C			M I
	;;
}{.mmi;	(p16)	ld8	r32 = [tp], 8		C			M
	(p50)	cmpeqor	p40, p0 = -1, r50	C cyout hi		M I
	(p17)	shrp	r16 = r33, r37, 63	C critical		I0
}{.mmi;	(p21)	getfsig	r42 = f37		C lo			M2
	(p23)	st8	[rp] = r44, 8		C lo			M23
	(p50)	add	r50 = 1, r50		C			M I
	;;
}		br.ctop.sptk.few.clr L(top)	C			B
dnl *** MAIN LOOP END ***
	;;
L(end):
 {.mmi;		nop	4711
	(p41)	add	r44 = 1, r44		C			M I
		shr.u	r48 = r39, 63		C			I0
	;;
}{.mmi;		st8	[rp] = r51, 8		C			M23
	(p41)	cmpequc	p6, p0 = 0, r44		C			M I
		add	r50 = r41, r48		C			M I
	;;
}{.mmi;		st8	[rp] = r44, 8		C			M23
	(p6)	add	r50 = 1, r50		C			M I
		mov	ar.lc = r3		C			I0
	;;
}{.mii;		st8	[rp] = r50		C			M23
		mov	ar.ec = r9		C			I0
		mov	pr = r10		C			I0
	;;
}{.mib;		nop	4711
		mov	ar.pfs = r2		C			I0
		br.ret.sptk.many b0		C			B
}
EPILOGUE()
