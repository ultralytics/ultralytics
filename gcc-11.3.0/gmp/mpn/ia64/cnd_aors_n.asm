dnl  IA-64 mpn_cnd_add_n/mpn_cnd_sub_n.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2013 Free Software Foundation, Inc.

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
C Itanium 2:    1.5

C INPUT PARAMETERS
define(`cnd', `r32')
define(`rp',  `r33')
define(`up',  `r34')
define(`vp',  `r35')
define(`n',   `r36')

ifdef(`OPERATION_cnd_add_n',`
  define(ADDSUB,	add)
  define(CND,		ltu)
  define(INCR,		1)
  define(LIM,		-1)
  define(func,    mpn_cnd_add_n)
')
ifdef(`OPERATION_cnd_sub_n',`
  define(ADDSUB,	sub)
  define(CND,		gtu)
  define(INCR,		-1)
  define(LIM,		0)
  define(func,    mpn_cnd_sub_n)
')

define(PFDIST, 160)

C Some useful aliases for registers we use
define(`u0',`r14') define(`u1',`r15') define(`u2',`r16') define(`u3',`r17')
define(`x0',`r20') define(`x1',`r21') define(`x2',`r22') define(`x3',`r23')
define(`v0',`r24') define(`v1',`r25') define(`v2',`r26') define(`v3',`r27')
define(`w0',`r28') define(`w1',`r29') define(`w2',`r30') define(`w3',`r31')
define(`up1',`up') define(`up2',`r8') define(`upadv',`r1')
define(`vp1',`vp') define(`vp2',`r9') define(`vpadv',`r11')
define(`rp1',`rp') define(`rp2',`r10')

MULFUNC_PROLOGUE(mpn_cnd_add_n mpn_cnd_sub_n)

ASM_START()
PROLOGUE(func)
	.prologue
	.save	ar.lc, r2
	.body
ifdef(`HAVE_ABI_32',`
	addp4	rp = 0, rp		C				M I
	addp4	up = 0, up		C				M I
	nop.i	0
	addp4	vp = 0, vp		C				M I
	nop.m	0
	zxt4	n = n			C				I
	;;
')
 {.mmi;	and	r3 = 3, n		C				M I
	add	n = -1, n		C				M I
	mov	r2 = ar.lc		C				I0
}{.mmi;	cmp.ne	p6, p7 = 0, cnd		C				M I
	add	vp2 = 8, vp		C				M I
	add	up2 = 8, up		C				M I
	;;
}{.mmi;	add	upadv = PFDIST, up	C				M I
	add	vpadv = PFDIST, vp	C				M I
	shr.u	n = n, 2		C				I0
	.pred.rel "mutex", p6, p7
}{.mmi;	add	rp2 = 8, rp		C				M I
   (p6)	mov	cnd = -1		C				M I
   (p7)	mov	cnd = 0			C				M I
	;;
}	cmp.eq	p9, p0 = 1, r3		C				M I
	cmp.eq	p7, p0 = 2, r3		C				M I
	cmp.eq	p8, p0 = 3, r3		C				M I
   (p9)	br	L(b1)			C				B
   (p7)	br	L(b2)			C				B
   (p8)	br	L(b3)			C				B
	;;
L(b0):
 {.mmi;	ld8	v2 = [vp1], 16		C				M01
	ld8	v3 = [vp2], 16		C				M01
	mov	ar.lc = n		C				I0
	;;
}	ld8	u2 = [up1], 16		C				M01
	ld8	u3 = [up2], 16		C				M01
	and	x2 = v2, cnd		C				M I
	and	x3 = v3, cnd		C				M I
	;;
	ADDSUB	w2 = u2, x2		C				M I
	ADDSUB	w3 = u3, x3		C				M I
	;;
	ld8	v0 = [vp1], 16		C				M01
	ld8	v1 = [vp2], 16		C				M01
	cmp.CND	p8, p0 = w2, u2		C				M I
	cmp.CND	p9, p0 = w3, u3		C				M I
	br	L(lo0)

L(b1):	ld8	v1 = [vp1], 8		C				M01
	add	vp2 = 8, vp2		C				M I
	add	rp2 = 8, rp2		C				M I
	;;
	ld8	u1 = [up1], 8		C				M01
	add	up2 = 8, up2		C				M I
	and	x1 = v1, cnd		C				M I
	;;
	ADDSUB	w1 = u1, x1		C				M I
	cmp.ne	p10, p0 = 0, n
	add	n = -1, n
	;;
	cmp.CND	p7, p0 = w1, u1		C				M I
	st8	[rp1] = w1, 8		C				M23
  (p10)	br	L(b0)
	;;
	mov	r8 = 0			C				M I
	br	L(e1)

L(b3):	ld8	v3 = [vp1], 8		C				M01
	add	vp2 = 8, vp2		C				M I
	add	rp2 = 8, rp2		C				M I
	;;
	ld8	u3 = [up1], 8		C				M01
	add	up2 = 8, up2		C				M I
	and	x3 = v3, cnd		C				M I
	;;
	ADDSUB	w3 = u3, x3		C				M I
	;;
	cmp.CND	p9, p0 = w3, u3		C				M I
	st8	[rp1] = w3, 8		C				M23
	C fall through

L(b2):
 {.mmi;	ld8	v0 = [vp1], 16		C				M01
	ld8	v1 = [vp2], 16		C				M01
	mov	ar.lc = n		C				I0
	;;
}	ld8	u0 = [up1], 16		C				M01
	ld8	u1 = [up2], 16		C				M01
	and	x0 = v0, cnd		C				M I
	and	x1 = v1, cnd		C				M I
	;;
	ADDSUB	w0 = u0, x0		C				M I
	ADDSUB	w1 = u1, x1		C				M I
	br.cloop.dptk	L(gt2)		C				B
	;;
	cmp.CND	p6, p0 = w0, u0		C				M I
	br		L(e2)		C				B
L(gt2):
	ld8	v2 = [vp1], 16		C				M01
	ld8	v3 = [vp2], 16		C				M01
	cmp.CND	p6, p0 = w0, u0		C				M I
	cmp.CND	p7, p0 = w1, u1		C				M I
	br		L(lo2)		C				B


C *** MAIN LOOP START ***
C	ALIGN(32)
L(top):
 {.mmi;	ld8	v2 = [vp1], 16		C				M01
	ld8	v3 = [vp2], 16		C				M01
	cmp.CND	p6, p0 = w0, u0		C				M I
}{.mmi;	st8	[rp1] = w2, 16		C				M23
	st8	[rp2] = w3, 16		C				M23
	cmp.CND	p7, p0 = w1, u1		C				M I
	;;
}
L(lo2):
 {.mmi;	ld8	u2 = [up1], 16		C				M01
	ld8	u3 = [up2], 16		C				M01
   (p9)	cmpeqor	p6, p0 = LIM, w0	C				M I
}{.mmi;	and	x2 = v2, cnd		C				M I
	and	x3 = v3, cnd		C				M I
   (p9)	add	w0 = INCR, w0		C				M I
	;;
}{.mmi;	ADDSUB	w2 = u2, x2		C				M I
   (p6)	cmpeqor	p7, p0 = LIM, w1	C				M I
   (p6)	add	w1 = INCR, w1		C				M I
}{.mmi;	ADDSUB	w3 = u3, x3		C				M I
	lfetch	[upadv], 32
	nop	0
	;;
}{.mmi;	ld8	v0 = [vp1], 16		C				M01
	ld8	v1 = [vp2], 16		C				M01
	cmp.CND	p8, p0 = w2, u2		C				M I
}{.mmi;	st8	[rp1] = w0, 16		C				M23
	st8	[rp2] = w1, 16		C				M23
	cmp.CND	p9, p0 = w3, u3		C				M I
	;;
}
L(lo0):
 {.mmi;	ld8	u0 = [up1], 16		C				M01
	ld8	u1 = [up2], 16		C				M01
   (p7)	cmpeqor	p8, p0 = LIM, w2	C				M I
}{.mmi;	and	x0 = v0, cnd		C				M I
	and	x1 = v1, cnd		C				M I
   (p7)	add	w2 = INCR, w2		C				M I
	;;
}{.mmi;	ADDSUB	w0 = u0, x0		C				M I
   (p8)	cmpeqor	p9, p0 = LIM, w3	C				M I
   (p8)	add	w3 = INCR, w3		C				M I
}{.mmb;	ADDSUB	w1 = u1, x1		C				M I
	lfetch	[vpadv], 32
	br.cloop.dptk	L(top)		C				B
	;;
}
C *** MAIN LOOP END ***


L(end):
 {.mmi;	st8	[rp1] = w2, 16		C				M23
	st8	[rp2] = w3, 16		C				M23
	cmp.CND	p6, p0 = w0, u0		C				M I
	;;
}
L(e2):
 {.mmi;	cmp.CND	p7, p0 = w1, u1		C				M I
   (p9)	cmpeqor	p6, p0 = LIM, w0	C				M I
   (p9)	add	w0 = INCR, w0		C				M I
	;;
}{.mmi;	mov	r8 = 0			C				M I
   (p6)	cmpeqor	p7, p0 = LIM, w1	C				M I
   (p6)	add	w1 = INCR, w1		C				M I
	;;
}{.mmi;	st8	[rp1] = w0, 16		C				M23
	st8	[rp2] = w1, 16		C				M23
	mov	ar.lc = r2		C				I0
}
L(e1):
 {.mmb;	nop	0
   (p7)	mov	r8 = 1			C				M I
	br.ret.sptk.many b0		C				B
}
EPILOGUE()
ASM_END()
