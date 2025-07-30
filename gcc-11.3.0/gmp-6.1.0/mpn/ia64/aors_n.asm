dnl  IA-64 mpn_add_n/mpn_sub_n -- mpn addition and subtraction.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2003-2005, 2010, 2011 Free Software Foundation, Inc.

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
C Itanium:      2.67
C Itanium 2:    1.25

C TODO
C  * Consider using special code for small n, using something like
C    "switch (8 * (n >= 8) + (n mod 8))" to enter it and feed-in code.
C  * The non-nc code was trimmed cycle for cycle to its current state.  It is
C    probably hard to save more that an odd cycle there.  The nc code is much
C    cruder (since tune/speed doesn't have any applicable direct measurements).
C  * Without the nc entry points, this becomes around 1800 bytes of object
C    code; the nc code adds over 1000 bytes.  We should perhaps sacrifice a
C    few cycles for the non-nc code and let it fall into the nc code.

C INPUT PARAMETERS
define(`rp', `r32')
define(`up', `r33')
define(`vp', `r34')
define(`n',  `r35')
define(`cy', `r36')

ifdef(`OPERATION_add_n',`
  define(ADDSUB,	add)
  define(CND,		ltu)
  define(INCR,		1)
  define(LIM,		-1)
  define(LIM2,		0)
  define(func,    mpn_add_n)
  define(func_nc, mpn_add_nc)
')
ifdef(`OPERATION_sub_n',`
  define(ADDSUB,	sub)
  define(CND,		gtu)
  define(INCR,		-1)
  define(LIM,		0)
  define(LIM2,		-1)
  define(func,    mpn_sub_n)
  define(func_nc, mpn_sub_nc)
')

define(PFDIST, 500)

C Some useful aliases for registers we use
define(`u0',`r14') define(`u1',`r15') define(`u2',`r16') define(`u3',`r17')
define(`v0',`r24') define(`v1',`r25') define(`v2',`r26') define(`v3',`r27')
define(`w0',`r28') define(`w1',`r29') define(`w2',`r30') define(`w3',`r31')
define(`rpx',`r3')
define(`upadv',`r20') define(`vpadv',`r21')

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc mpn_sub_n mpn_sub_nc)

ASM_START()
PROLOGUE(func_nc)
	.prologue
	.save	ar.lc, r2
	.body
ifdef(`HAVE_ABI_32',`
		addp4	rp = 0, rp		C			M I
		addp4	up = 0, up		C			M I
		nop.i	0
		addp4	vp = 0, vp		C			M I
		nop.m	0
		zxt4	n = n			C			I
	;;
')

 {.mmi;		ld8	r11 = [vp], 8		C			M01
		ld8	r10 = [up], 8		C			M01
		mov	r2 = ar.lc		C			I0
}{.mmi;		and	r14 = 7, n		C			M I
		cmp.lt	p15, p14 = 8, n		C			M I
		add	n = -6, n		C			M I
	;;
}{.mmi;		add	upadv = PFDIST, up	C Merging these lines into the feed-in
		add	vpadv = PFDIST, vp	C code could save a cycle per call at
		mov	r23 = cy		C the expense of code size.
	;;
}{.mmi;		cmp.eq	p6, p0 = 1, r14		C			M I
		cmp.eq	p7, p0 = 2, r14		C			M I
		cmp.eq	p8, p0 = 3, r14		C			M I
}{.bbb;	(p6)	br.dptk	.Lc001			C			B
	(p7)	br.dptk	.Lc010			C			B
	(p8)	br.dptk	.Lc011			C			B
	;;
}{.mmi;		cmp.eq	p9, p0 = 4, r14		C			M I
		cmp.eq	p10, p0 = 5, r14	C			M I
		cmp.eq	p11, p0 = 6, r14	C			M I
}{.bbb;	(p9)	br.dptk	.Lc100			C			B
	(p10)	br.dptk	.Lc101			C			B
	(p11)	br.dptk	.Lc110			C			B
	;;
}{.mmi;		ld8	r19 = [vp], 8		C			M01
		ld8	r18 = [up], 8		C			M01
		cmp.ne	p13, p0 = 0, cy		C copy cy to p13	M I
}{.mmb;		cmp.eq	p12, p0 = 7, r14	C			M I
		nop	0
	(p12)	br.dptk	.Lc111			C			B
	;;
}

.Lc000:
 {.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
	;;
}{.mmi;		add	vpadv = PFDIST, vp	C			M I
		ld8	v0 = [vp], 8		C			M01
		mov	ar.lc = n		C			I0
}{.mmi;		ld8	u0 = [up], 8		C			M01
		ADDSUB	w1 = r10, r11		C			M I
		nop	0
	;;
}{.mmi;		add	upadv = PFDIST, up	C			M I
		ld8	v1 = [vp], 8		C			M01
		cmp.CND	p7, p0 = w1, r10	C			M I
}{.mmi;		ld8	u1 = [up], 8		C			M01
		ADDSUB	w2 = r18, r19		C			M I
		add	rpx = 8, rp		C			M I
	;;
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		cmp.CND	p8, p0 = w2, r18	C			M I
	(p13)	cmpeqor	p7, p0 = LIM, w1	C			M I
}{.mmi;		ld8	u2 = [up], 8		C			M01
	(p13)	add	w1 = INCR, w1		C			M I
		ADDSUB	w3 = u3, v3		C			M I
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		cmp.CND	p9, p0 = w3, u3		C			M I
	(p7)	cmpeqor	p8, p0 = LIM, w2	C			M I
}{.mmb;		ld8	u3 = [up], 8		C			M01
	(p7)	add	w2 = INCR, w2		C			M I
		br	L(m0)
}

.Lc001:
 {.mmi;	(p15)	ld8	v1 = [vp], 8		C			M01
	(p15)	ld8	u1 = [up], 8		C			M01
		ADDSUB	w0 = r10, r11		C			M I
}{.mmb;		nop	0
		nop	0
	(p15)	br	L(0)
	;;
}{.mmi;		cmp.ne	p9, p0 = 0, r23		C			M I
		mov	r8 = 0
		cmp.CND	p6, p0 = w0, r10	C			M I
	;;
}{.mmb;	(p9)	cmpeqor	p6, p0 = LIM, w0	C			M I
	(p9)	add	w0 = INCR, w0		C			M I
		br	L(cj1)			C			B
}
L(0):
 {.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		mov	ar.lc = n		C			I0
}{.mmi;		nop	0
		cmp.ne	p9, p0 = 0, r23		C			M I
		nop	0
	;;
}{.mmi;		ld8	v0 = [vp], 8		C			M01
		cmp.CND	p6, p0 = w0, r10	C			M I
		add	rpx = 16, rp		C			M I
}{.mmb;		ld8	u0 = [up], 8		C			M01
		ADDSUB	w1 = u1, v1		C			M I
		br	L(c1)			C			B
}

.Lc010:
 {.mmi;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
		mov	r8 = 0			C			M I
}{.mmb;		ADDSUB	w3 = r10, r11		C			M I
		cmp.ne	p8, p0 = 0, r23		C			M I
	(p15)	br	L(1)			C			B
	;;
}{.mmi;		cmp.CND	p9, p0 = w3, r10	C			M I
		ADDSUB	w0 = u0, v0		C			M I
	(p8)	add	w3 = INCR, w3		C			M I
	;;
}{.mmb;		cmp.CND	p6, p0 = w0, u0		C			M I
	(p8)	cmpeqor	p9, p0 = LIM2, w3	C			M I
		br	L(cj2)			C			B
}
L(1):
 {.mmi;		ld8	v1 = [vp], 8		C			M01
		ld8	u1 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
	;;
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		mov	ar.lc = n		C			I0
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		cmp.CND	p9, p0 = w3, r10	C			M I
	;;
}{.mmi;	(p8)	cmpeqor	p9, p0 = LIM, w3	C			M I
	(p8)	add	w3 = INCR, w3		C			M I
		ADDSUB	w0 = u0, v0		C			M I
}{.mmb;		add	rpx = 24, rp		C			M I
		nop	0
		br	L(m23)			C			B
}

.Lc011:
 {.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
}{.mmi;		ADDSUB	w2 = r10, r11		C			M I
		cmp.ne	p7, p0 = 0, r23		C			M I
		nop	0
	;;
}{.mmb;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
	(p15)	br	L(2)			C			B
}{.mmi;		cmp.CND	p8, p0 = w2, r10	C			M I
		ADDSUB	w3 = u3, v3		C			M I
		nop	0
	;;
}{.mmb;	(p7)	cmpeqor	p8, p0 = LIM, w2	C			M I
	(p7)	add	w2 = INCR, w2		C			M I
		br	L(cj3)			C			B
}
L(2):
 {.mmi;		ld8	v1 = [vp], 8		C			M01
		ld8	u1 = [up], 8		C			M01
		ADDSUB	w3 = u3, v3		C			M I
	;;
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		cmp.CND	p8, p0 = w2, r10	C			M I
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		cmp.CND	p9, p0 = w3, u3		C			M I
		mov	ar.lc = n		C			I0
}{.mmi;		ld8	u3 = [up], 8		C			M01
	(p7)	cmpeqor	p8, p0 = LIM, w2	C			M I
	(p7)	add	w2 = INCR, w2		C			M I
	;;
}{.mmi;		add	rpx = 32, rp		C			M I
		st8	[rp] = w2, 8		C			M23
	(p8)	cmpeqor	p9, p0 = LIM, w3	C			M I
}{.mmb;	(p8)	add	w3 = INCR, w3		C			M I
		ADDSUB	w0 = u0, v0		C			M I
		br	L(m23)
}

.Lc100:
 {.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
}{.mmi;		ADDSUB	w1 = r10, r11		C			M I
		nop	0
		nop	0
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		add	rpx = 8, rp		C			M I
}{.mmi;		cmp.ne	p6, p0 = 0, r23		C			M I
		cmp.CND	p7, p0 = w1, r10	C			M I
		nop	0
	;;
}{.mmi;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
		ADDSUB	w2 = u2, v2		C			M I
}{.mmb;	(p6)	cmpeqor	p7, p0 = LIM, w1	C			M I
	(p6)	add	w1 = INCR, w1		C			M I
	(p14)	br	L(cj4)
	;;
}{.mmi;		ld8	v1 = [vp], 8		C			M01
		ld8	u1 = [up], 8		C			M01
		mov	ar.lc = n		C			I0
	;;
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		cmp.CND	p8, p0 = w2, u2		C			M I
		nop	0
}{.mmi;		ld8	u2 = [up], 8		C			M01
		nop	0
		ADDSUB	w3 = u3, v3		C			M I
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		cmp.CND	p9, p0 = w3, u3		C			M I
	(p7)	cmpeqor	p8, p0 = LIM, w2	C			M I
}{.mmb;		ld8	u3 = [up], 8		C			M01
	(p7)	add	w2 = INCR, w2		C			M I
		br	L(m4)
}

.Lc101:
 {.mmi;		ld8	v1 = [vp], 8		C			M01
		ld8	u1 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
	;;
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		mov	ar.lc = n		C			I0
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		ADDSUB	w0 = r10, r11		C			M I
}{.mmi;		cmp.ne	p9, p0 = 0, r23		C			M I
		add	rpx = 16, rp		C			M I
		nop	0
	;;
}{.mmi;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
		cmp.CND	p6, p0 = w0, r10	C			M I
}{.mbb;		ADDSUB	w1 = u1, v1		C			M I
	(p15)	br	L(c5)			C			B
		br	L(end)			C			B
}

.Lc110:
 {.mmi;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
	;;
}{.mmi;		add	upadv = PFDIST, up	C			M I
		add	vpadv = PFDIST, vp	C			M I
		mov	ar.lc = n		C			I0
}{.mmi;		ld8	v1 = [vp], 8		C			M01
		ld8	u1 = [up], 8		C			M01
		ADDSUB	w3 = r10, r11		C			M I
	;;
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		ADDSUB	w0 = u0, v0		C			M I
}{.mmi;		cmp.CND	p9, p0 = w3, r10	C			M I
		cmp.ne	p8, p0 = 0, r23		C			M I
		add	rpx = 24, rp		C			M I
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		nop	0
}{.mmb;	(p8)	cmpeqor	p9, p0 = LIM, w3	C			M I
	(p8)	add	w3 = INCR, w3		C			M I
		br	L(m67)			C			B
}

.Lc111:
 {.mmi;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
	;;
}{.mmi;		add	upadv = PFDIST, up	C			M I
		ld8	v1 = [vp], 8		C			M01
		mov	ar.lc = n		C			I0
}{.mmi;		ld8	u1 = [up], 8		C			M01
		ADDSUB	w2 = r10, r11		C			M I
		nop	0
	;;
}{.mmi;		add	vpadv = PFDIST, vp	C			M I
		ld8	v2 = [vp], 8		C			M01
		cmp.CND	p8, p0 = w2, r10	C			M I
}{.mmi;		ld8	u2 = [up], 8		C			M01
		ADDSUB	w3 = r18, r19		C			M I
		nop	0
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		cmp.CND	p9, p0 = w3, r18	C			M I
	(p13)	cmpeqor	p8, p0 = LIM, w2	C			M I
}{.mmi;		ld8	u3 = [up], 8		C			M01
	(p13)	add	w2 = INCR, w2		C			M I
		nop	0
	;;
}{.mmi;		add	rpx = 32, rp		C			M I
		st8	[rp] = w2, 8		C			M23
	(p8)	cmpeqor	p9, p0 = LIM, w3	C			M I
}{.mmb;	(p8)	add	w3 = INCR, w3		C			M I
		ADDSUB	w0 = u0, v0		C			M I
		br	L(m67)
}
EPILOGUE()

PROLOGUE(func)
	.prologue
	.save	ar.lc, r2
	.body
ifdef(`HAVE_ABI_32',`
		addp4	rp = 0, rp		C			M I
		addp4	up = 0, up		C			M I
		nop.i	0
		addp4	vp = 0, vp		C			M I
		nop.m	0
		zxt4	n = n			C			I
	;;
')

 {.mmi;		ld8	r11 = [vp], 8		C			M01
		ld8	r10 = [up], 8		C			M01
		mov	r2 = ar.lc		C			I0
}{.mmi;		and	r14 = 7, n		C			M I
		cmp.lt	p15, p14 = 8, n		C			M I
		add	n = -6, n		C			M I
	;;
}{.mmi;		cmp.eq	p6, p0 = 1, r14		C			M I
		cmp.eq	p7, p0 = 2, r14		C			M I
		cmp.eq	p8, p0 = 3, r14		C			M I
}{.bbb;	(p6)	br.dptk	.Lb001			C			B
	(p7)	br.dptk	.Lb010			C			B
	(p8)	br.dptk	.Lb011			C			B
	;;
}{.mmi;		cmp.eq	p9, p0 = 4, r14		C			M I
		cmp.eq	p10, p0 = 5, r14	C			M I
		cmp.eq	p11, p0 = 6, r14	C			M I
}{.bbb;	(p9)	br.dptk	.Lb100			C			B
	(p10)	br.dptk	.Lb101			C			B
	(p11)	br.dptk	.Lb110			C			B
	;;
}{.mmi;		ld8	r19 = [vp], 8		C			M01
		ld8	r18 = [up], 8		C			M01
		cmp.ne	p13, p0 = r0, r0	C clear "CF"		M I
}{.mmb;		cmp.eq	p12, p0 = 7, r14	C			M I
		mov	r23 = 0			C			M I
	(p12)	br.dptk	.Lb111			C			B
	;;
}

.Lb000:
 {.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
	;;
}{.mmi;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
		ADDSUB	w1 = r10, r11		C			M I
	;;
}{.mmi;		ld8	v1 = [vp], 8		C			M01
		cmp.CND	p7, p0 = w1, r10	C			M I
		mov	ar.lc = n		C			I0
}{.mmi;		ld8	u1 = [up], 8		C			M01
		ADDSUB	w2 = r18, r19		C			M I
		add	rpx = 8, rp		C			M I
	;;
}{.mmi;		add	upadv = PFDIST, up
		add	vpadv = PFDIST, vp
		cmp.CND	p8, p0 = w2, r18	C			M I
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		ADDSUB	w3 = u3, v3		C			M I
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		cmp.CND	p9, p0 = w3, u3		C			M I
	(p7)	cmpeqor	p8, p0 = LIM, w2	C			M I
}{.mmb;		ld8	u3 = [up], 8		C			M01
	(p7)	add	w2 = INCR, w2		C			M I
		br	L(m0)			C			B
}

	ALIGN(32)
.Lb001:
 {.mmi;		ADDSUB	w0 = r10, r11		C			M I
	(p15)	ld8	v1 = [vp], 8		C			M01
		mov	r8 = 0			C			M I
	;;
}{.mmb;		cmp.CND	p6, p0 = w0, r10	C			M I
	(p15)	ld8	u1 = [up], 8		C			M01
	(p14)	br	L(cj1)			C			B
	;;
}{.mmi;		add	upadv = PFDIST, up
		add	vpadv = PFDIST, vp
		shr.u	n = n, 3		C			I0
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		cmp.CND	p6, p0 = w0, r10	C			M I
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		mov	ar.lc = n		C			I0
	;;
}{.mmi;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
		ADDSUB	w1 = u1, v1		C			M I
	;;
}{.mmi;		ld8	v1 = [vp], 8		C			M01
		cmp.CND	p7, p0 = w1, u1		C			M I
		ADDSUB	w2 = u2, v2		C			M I
}{.mmb;		ld8	u1 = [up], 8		C			M01
		add	rpx = 16, rp		C			M I
		br	L(m1)			C			B
}

	ALIGN(32)
.Lb010:
 {.mmi;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
}{.mmb;		ADDSUB	w3 = r10, r11		C			M I
		nop	0
	(p15)	br	L(gt2)			C			B
	;;
}{.mmi;		cmp.CND	p9, p0 = w3, r10	C			M I
		ADDSUB	w0 = u0, v0		C			M I
		mov	r8 = 0			C			M I
	;;
}{.mmb;		nop	0
		cmp.CND	p6, p0 = w0, u0		C			M I
		br	L(cj2)			C			B
}
L(gt2):
 {.mmi;		ld8	v1 = [vp], 8		C			M01
		ld8	u1 = [up], 8		C			M01
		nop	0
	;;
}{.mmi;		add	upadv = PFDIST, up
		add	vpadv = PFDIST, vp
		mov	ar.lc = n		C			I0
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		nop	0
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		cmp.CND	p9, p0 = w3, r10	C			M I
		ADDSUB	w0 = u0, v0		C			M I
}{.mmb;		ld8	u3 = [up], 8		C			M01
		add	rpx = 24, rp		C			M I
		br	L(m23)			C			B
}

	ALIGN(32)
.Lb011:
 {.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		ADDSUB	w2 = r10, r11		C			M I
	;;
}{.mmb;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
	(p15)	br	L(3)			C			B
}{.mmb;		cmp.CND	p8, p0 = w2, r10	C			M I
		ADDSUB	w3 = u3, v3		C			M I
		br	L(cj3)			C			B
}
L(3):
 {.mmi;		ld8	v1 = [vp], 8		C			M01
		ld8	u1 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
	;;
}{.mmi;		add	upadv = PFDIST, up
		add	vpadv = PFDIST, vp
		ADDSUB	w3 = u3, v3		C			M I
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		cmp.CND	p8, p0 = w2, r10	C			M I
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		cmp.CND	p9, p0 = w3, u3		C			M I
		mov	ar.lc = n		C			I0
}{.mmi;		ld8	u3 = [up], 8		C			M01
		nop	0
		nop	0
	;;
}{.mmi;		add	rpx = 32, rp		C			M I
		st8	[rp] = w2, 8		C			M23
	(p8)	cmpeqor	p9, p0 = LIM, w3	C			M I
}{.mmb;	(p8)	add	w3 = INCR, w3		C			M I
		ADDSUB	w0 = u0, v0		C			M I
		br	L(m23)			C			B
}

	ALIGN(32)
.Lb100:
 {.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		ADDSUB	w1 = r10, r11		C			M I
	;;
}{.mmi;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
		cmp.CND	p7, p0 = w1, r10	C			M I
}{.mmb;		nop	0
		ADDSUB	w2 = u2, v2		C			M I
	(p14)	br	L(cj4)			C			B
	;;
}
L(gt4):
 {.mmi;		add	upadv = PFDIST, up
		add	vpadv = PFDIST, vp
		mov	ar.lc = n		C			I0
}{.mmi;		ld8	v1 = [vp], 8		C			M01
		ld8	u1 = [up], 8		C			M01
		nop	0
	;;
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		cmp.CND	p8, p0 = w2, u2		C			M I
		nop	0
}{.mmi;		ld8	u2 = [up], 8		C			M01
		ADDSUB	w3 = u3, v3		C			M I
		add	rpx = 8, rp		C			M I
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		cmp.CND	p9, p0 = w3, u3		C			M I
	(p7)	cmpeqor	p8, p0 = LIM, w2	C			M I
}{.mmb;		ld8	u3 = [up], 8		C			M01
	(p7)	add	w2 = INCR, w2		C			M I
		br	L(m4)			C			B
}

	ALIGN(32)
.Lb101:
 {.mmi;		ld8	v1 = [vp], 8		C			M01
		ld8	u1 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
	;;
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		ADDSUB	w0 = r10, r11		C			M I
	;;
}{.mmi;		add	upadv = PFDIST, up
		add	vpadv = PFDIST, vp
		add	rpx = 16, rp		C			M I
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		nop	0
	;;
}{.mmi;		ld8	v0 = [vp], 8		C			M01
		cmp.CND	p6, p0 = w0, r10	C			M I
		nop	0
}{.mmb;		ld8	u0 = [up], 8		C			M01
		ADDSUB	w1 = u1, v1		C			M I
	(p14)	br	L(cj5)			C			B
	;;
}
L(gt5):
 {.mmi;		ld8	v1 = [vp], 8		C			M01
		cmp.CND	p7, p0 = w1, u1		C			M I
		mov	ar.lc = n		C			I0
}{.mmb;		ld8	u1 = [up], 8		C			M01
		ADDSUB	w2 = u2, v2		C			M I
		br	L(m5)			C			B
}

	ALIGN(32)
.Lb110:
 {.mmi;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
	;;
}{.mmi;		ld8	v1 = [vp], 8		C			M01
		ld8	u1 = [up], 8		C			M01
		ADDSUB	w3 = r10, r11		C			M I
	;;
}{.mmi;		add	upadv = PFDIST, up
		add	vpadv = PFDIST, vp
		mov	ar.lc = n		C			I0
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		ld8	u2 = [up], 8		C			M01
		nop	0
	;;
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		cmp.CND	p9, p0 = w3, r10	C			M I
		ADDSUB	w0 = u0, v0		C			M I
}{.mmb;		ld8	u3 = [up], 8		C			M01
		add	rpx = 24, rp		C			M I
		br	L(m67)			C			B
}

	ALIGN(32)
.Lb111:
 {.mmi;		ld8	v0 = [vp], 8		C			M01
		ld8	u0 = [up], 8		C			M01
		shr.u	n = n, 3		C			I0
	;;
}{.mmi;		ld8	v1 = [vp], 8		C			M01
		ld8	u1 = [up], 8		C			M01
		ADDSUB	w2 = r10, r11		C			M I
	;;
}{.mmi;		ld8	v2 = [vp], 8		C			M01
		cmp.CND	p8, p0 = w2, r10	C			M I
		mov	ar.lc = n		C			I0
}{.mmi;		ld8	u2 = [up], 8		C			M01
		ADDSUB	w3 = r18, r19		C			M I
		nop	0
	;;
}{.mmi;		add	upadv = PFDIST, up
		add	vpadv = PFDIST, vp
		nop	0
}{.mmi;		ld8	v3 = [vp], 8		C			M01
		ld8	u3 = [up], 8		C			M01
		cmp.CND	p9, p0 = w3, r18	C			M I
	;;
}{.mmi;		add	rpx = 32, rp		C			M I
		st8	[rp] = w2, 8		C			M23
	(p8)	cmpeqor	p9, p0 = LIM, w3	C			M I
}{.mmb;	(p8)	add	w3 = INCR, w3		C			M I
		ADDSUB	w0 = u0, v0		C			M I
		br	L(m67)			C			B
}

C *** MAIN LOOP START ***
	ALIGN(32)
L(top):
L(c5):		ld8	v1 = [vp], 8		C			M01
		cmp.CND	p7, p0 = w1, u1		C			M I
	(p9)	cmpeqor	p6, p0 = LIM, w0	C			M I
		ld8	u1 = [up], 8		C			M01
	(p9)	add	w0 = INCR, w0		C			M I
		ADDSUB	w2 = u2, v2		C			M I
	;;
L(m5):		ld8	v2 = [vp], 8		C			M01
		cmp.CND	p8, p0 = w2, u2		C			M I
	(p6)	cmpeqor	p7, p0 = LIM, w1	C			M I
		ld8	u2 = [up], 8		C			M01
	(p6)	add	w1 = INCR, w1		C			M I
		ADDSUB	w3 = u3, v3		C			M I
	;;
		st8	[rp] = w0, 8		C			M23
		ld8	v3 = [vp], 8		C			M01
		cmp.CND	p9, p0 = w3, u3		C			M I
	(p7)	cmpeqor	p8, p0 = LIM, w2	C			M I
		ld8	u3 = [up], 8		C			M01
	(p7)	add	w2 = INCR, w2		C			M I
	;;
L(m4):		st8	[rp] = w1, 16		C			M23
		st8	[rpx] = w2, 32		C			M23
	(p8)	cmpeqor	p9, p0 = LIM, w3	C			M I
		lfetch	[upadv], 64
	(p8)	add	w3 = INCR, w3		C			M I
		ADDSUB	w0 = u0, v0		C			M I
	;;
L(m23):		st8	[rp] = w3, 8		C			M23
		ld8	v0 = [vp], 8		C			M01
		cmp.CND	p6, p0 = w0, u0		C			M I
		ld8	u0 = [up], 8		C			M01
		ADDSUB	w1 = u1, v1		C			M I
		nop.b	0
	;;
L(c1):		ld8	v1 = [vp], 8		C			M01
		cmp.CND	p7, p0 = w1, u1		C			M I
	(p9)	cmpeqor	p6, p0 = LIM, w0	C			M I
		ld8	u1 = [up], 8		C			M01
	(p9)	add	w0 = INCR, w0		C			M I
		ADDSUB	w2 = u2, v2		C			M I
	;;
L(m1):		ld8	v2 = [vp], 8		C			M01
		cmp.CND	p8, p0 = w2, u2		C			M I
	(p6)	cmpeqor	p7, p0 = LIM, w1	C			M I
		ld8	u2 = [up], 8		C			M01
	(p6)	add	w1 = INCR, w1		C			M I
		ADDSUB	w3 = u3, v3		C			M I
	;;
		st8	[rp] = w0, 8		C			M23
		ld8	v3 = [vp], 8		C			M01
		cmp.CND	p9, p0 = w3, u3		C			M I
	(p7)	cmpeqor	p8, p0 = LIM, w2	C			M I
		ld8	u3 = [up], 8		C			M01
	(p7)	add	w2 = INCR, w2		C			M I
	;;
L(m0):		st8	[rp] = w1, 16		C			M23
		st8	[rpx] = w2, 32		C			M23
	(p8)	cmpeqor	p9, p0 = LIM, w3	C			M I
		lfetch	[vpadv], 64
	(p8)	add	w3 = INCR, w3		C			M I
		ADDSUB	w0 = u0, v0		C			M I
	;;
L(m67):		st8	[rp] = w3, 8		C			M23
		ld8	v0 = [vp], 8		C			M01
		cmp.CND	p6, p0 = w0, u0		C			M I
		ld8	u0 = [up], 8		C			M01
		ADDSUB	w1 = u1, v1		C			M I
		br.cloop.dptk	L(top)		C			B
	;;
C *** MAIN LOOP END ***

L(end):
 {.mmi;	(p9)	cmpeqor	p6, p0 = LIM, w0	C			M I
	(p9)	add	w0 = INCR, w0		C			M I
		mov	ar.lc = r2		C			I0
}
L(cj5):
 {.mmi;		cmp.CND	p7, p0 = w1, u1		C			M I
		ADDSUB	w2 = u2, v2		C			M I
		nop	0
	;;
}{.mmi;		st8	[rp] = w0, 8		C			M23
	(p6)	cmpeqor	p7, p0 = LIM, w1	C			M I
	(p6)	add	w1 = INCR, w1		C			M I
}
L(cj4):
 {.mmi;		cmp.CND	p8, p0 = w2, u2		C			M I
		ADDSUB	w3 = u3, v3		C			M I
		nop	0
	;;
}{.mmi;		st8	[rp] = w1, 8		C			M23
	(p7)	cmpeqor	p8, p0 = LIM, w2	C			M I
	(p7)	add	w2 = INCR, w2		C			M I
}
L(cj3):
 {.mmi;		cmp.CND	p9, p0 = w3, u3		C			M I
		ADDSUB	w0 = u0, v0		C			M I
		nop	0
	;;
}{.mmi;		st8	[rp] = w2, 8		C			M23
	(p8)	cmpeqor	p9, p0 = LIM, w3	C			M I
	(p8)	add	w3 = INCR, w3		C			M I
}{.mmi;		cmp.CND	p6, p0 = w0, u0		C			M I
		nop	0
		mov	r8 = 0			C			M I
	;;
}
L(cj2):
 {.mmi;		st8	[rp] = w3, 8		C			M23
	(p9)	cmpeqor	p6, p0 = LIM, w0	C			M I
	(p9)	add	w0 = INCR, w0		C			M I
	;;
}
L(cj1):
 {.mmb;		st8	[rp] = w0, 8		C			M23
	(p6)	mov	r8 = 1			C			M I
		br.ret.sptk.many b0		C			B
}
EPILOGUE()
ASM_END()
