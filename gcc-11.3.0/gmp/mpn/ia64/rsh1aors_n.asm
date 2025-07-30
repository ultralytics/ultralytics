dnl  IA-64 mpn_rsh1add_n/mpn_rsh1sub_n -- rp[] = (up[] +- vp[]) >> 1.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2003-2005 Free Software Foundation, Inc.

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
C Itanium:    2.5
C Itanium 2:  1.5

C TODO
C  * Rewrite function entry code using aorslsh1_n.asm style.
C  * Micro-optimize feed-in and wind-down code.

C INPUT PARAMETERS
define(`rp',`r32')
define(`up',`r33')
define(`vp',`r34')
define(`n',`r35')

ifdef(`OPERATION_rsh1add_n',`
  define(ADDSUB,       add)
  define(PRED,	       ltu)
  define(INCR,	       1)
  define(LIM,	       -1)
  define(func, mpn_rsh1add_n)
')
ifdef(`OPERATION_rsh1sub_n',`
  define(ADDSUB,       sub)
  define(PRED,	       gtu)
  define(INCR,	       -1)
  define(LIM,	       0)
  define(func, mpn_rsh1sub_n)
')

C Some useful aliases for registers we use
define(`u0',`r14') define(`u1',`r15') define(`u2',`r16') define(`u3',`r17')
define(`v0',`r18') define(`v1',`r19') define(`v2',`r20') define(`v3',`r21')
define(`w0',`r22') define(`w1',`r23') define(`w2',`r24') define(`w3',`r25')
define(`x0',`r26') define(`x1',`r9') define(`x2',`r30') define(`x3',`r31')

MULFUNC_PROLOGUE(mpn_rsh1add_n mpn_rsh1sub_n)

ASM_START()
PROLOGUE(func)
	.prologue
	.save	ar.lc, r2
	.body
ifdef(`HAVE_ABI_32',`
	addp4		rp = 0, rp		C			M I
	addp4		up = 0, up		C			M I
	addp4		vp = 0, vp		C			M I
	nop.m		0
	nop.m		0
	zxt4		n = n			C			I
	;;
')
 {.mmi;	ld8		r11 = [vp], 8		C			M01
	ld8		r10 = [up], 8		C			M01
	mov.i		r2 = ar.lc		C			I0
}{.mmi;	and		r14 = 3, n		C			M I
	cmp.lt		p15, p0 = 4, n		C			M I
	add		n = -4, n		C			M I
	;;
}{.mmi;	cmp.eq		p6, p0 = 1, r14		C			M I
	cmp.eq		p7, p0 = 2, r14		C			M I
	cmp.eq		p8, p0 = 3, r14		C			M I
}{.bbb
  (p6)	br.dptk		.Lb01			C			B
  (p7)	br.dptk		.Lb10			C			B
  (p8)	br.dptk		.Lb11			C			B
}

.Lb00:	ld8		v0 = [vp], 8		C			M01
	ld8		u0 = [up], 8		C			M01
	shr.u		n = n, 2		C			I0
	;;
	ld8		v1 = [vp], 8		C			M01
	ld8		u1 = [up], 8		C			M01
	ADDSUB		w3 = r10, r11		C			M I
	;;
	ld8		v2 = [vp], 8		C			M01
	ld8		u2 = [up], 8		C			M01
  (p15)	br.dpnt		.grt4			C			B
	;;

	cmp.PRED	p7, p0 = w3, r10	C			M I
	and		r8 = 1, w3		C			M I
	ADDSUB		w0 = u0, v0		C			M I
	;;
	cmp.PRED	p8, p0 = w0, u0		C			M I
	ADDSUB		w1 = u1, v1		C			M I
	;;
	cmp.PRED	p9, p0 = w1, u1		C			M I
   (p7)	cmp.eq.or	p8, p0 = LIM, w0	C			M I
   (p7)	add		w0 = INCR, w0		C			M I
	;;
	shrp		x3 = w0, w3, 1		C			I0
	ADDSUB		w2 = u2, v2		C			M I
   (p8)	cmp.eq.or	p9, p0 = LIM, w1	C			M I
   (p8)	add		w1 = INCR, w1		C			M I
	br		.Lcj4			C			B

.grt4:	ld8		v3 = [vp], 8		C			M01
	cmp.PRED	p7, p0 = w3, r10	C			M I
	ld8		u3 = [up], 8		C			M01
	and		r8 = 1, w3		C			M I
	;;
	ADDSUB		w0 = u0, v0		C			M I
	ld8		v0 = [vp], 8		C			M01
	add		n = -1, n
	;;
	cmp.PRED	p8, p0 = w0, u0		C			M I
	ld8		u0 = [up], 8		C			M01
	ADDSUB		w1 = u1, v1		C			M I
	;;
	ld8		v1 = [vp], 8		C			M01
	mov.i		ar.lc = n		C			I0
	cmp.PRED	p9, p0 = w1, u1		C			M I
	ld8		u1 = [up], 8		C			M01
   (p7)	cmp.eq.or	p8, p0 = LIM, w0	C			M I
   (p7)	add		w0 = INCR, w0		C			M I
	;;
	ADDSUB		w2 = u2, v2		C			M I
	ld8		v2 = [vp], 8		C			M01
	shrp		x3 = w0, w3, 1		C			I0
   (p8)	cmp.eq.or	p9, p0 = LIM, w1	C			M I
   (p8)	add		w1 = INCR, w1		C			M I
	br		.LL00			C			B


.Lb01:	ADDSUB		w2 = r10, r11		C			M I
	shr.u		n = n, 2		C			I0
  (p15)	br.dpnt		.grt1			C			B
	;;

	cmp.PRED	p6, p7 = w2, r10	C			M I
	shr.u		x2 = w2, 1		C			I0
	and		r8 = 1, w2		C			M I
	;;
   (p6)	dep		x2 = -1, x2, 63, 1	C			I0
	br		.Lcj1			C			B

.grt1:	ld8		v3 = [vp], 8		C			M01
	ld8		u3 = [up], 8		C			M01
	;;
	ld8		v0 = [vp], 8		C			M01
	ld8		u0 = [up], 8		C			M01
	mov.i		ar.lc = n		C FIXME swap with next	I0
	;;
	ld8		v1 = [vp], 8		C			M01
	ld8		u1 = [up], 8		C			M01
	;;
	ld8		v2 = [vp], 8		C			M01
	ld8		u2 = [up], 8		C			M01
	cmp.PRED	p6, p0 = w2, r10	C			M I
	and		r8 = 1, w2		C			M I
	ADDSUB		w3 = u3, v3		C			M I
	br.cloop.dptk	.grt5			C			B
	;;

	cmp.PRED	p7, p0 = w3, u3		C			M I
	;;
	ADDSUB		w0 = u0, v0		C			M I
   (p6)	cmp.eq.or	p7, p0 = LIM, w3	C			M I
   (p6)	add		w3 = INCR, w3		C			M I
	;;
	cmp.PRED	p8, p0 = w0, u0		C			M I
	shrp		x2 = w3, w2, 1		C			I0
	ADDSUB		w1 = u1, v1		C			M I
	;;
	cmp.PRED	p9, p0 = w1, u1		C			M I
   (p7)	cmp.eq.or	p8, p0 = LIM, w0	C			M I
   (p7)	add		w0 = INCR, w0		C			M I
	br		.Lcj5			C			B

.grt5:	ld8		v3 = [vp], 8		C			M01
	cmp.PRED	p7, p0 = w3, u3		C			M I
	ld8		u3 = [up], 8		C			M01
	;;
	ADDSUB		w0 = u0, v0		C			M I
	ld8		v0 = [vp], 8		C			M01
   (p6)	cmp.eq.or	p7, p0 = LIM, w3	C			M I
   (p6)	add		w3 = INCR, w3		C			M I
	;;
	cmp.PRED	p8, p0 = w0, u0		C			M I
	shrp		x2 = w3, w2, 1		C			I0
	ld8		u0 = [up], 8		C			M01
	ADDSUB		w1 = u1, v1		C			M I
	;;
	ld8		v1 = [vp], 8		C			M01
	cmp.PRED	p9, p0 = w1, u1		C			M I
	ld8		u1 = [up], 8		C			M01
   (p7)	cmp.eq.or	p8, p0 = LIM, w0	C			M I
   (p7)	add		w0 = INCR, w0		C			M I
	br		.LL01			C			B


.Lb10:	ld8		v2 = [vp], 8		C			M01
	ld8		u2 = [up], 8		C			M01
	shr.u		n = n, 2		C			I0
	ADDSUB		w1 = r10, r11		C			M I
  (p15)	br.dpnt		.grt2			C			B
	;;

	cmp.PRED	p9, p0 = w1, r10	C			M I
	and		r8 = 1, w1		C			M I
	ADDSUB		w2 = u2, v2		C			M I
	;;
	cmp.PRED	p6, p0 = w2, u2		C			M I
	;;
   (p9)	cmp.eq.or	p6, p0 = LIM, w2	C			M I
   (p9)	add		w2 = INCR, w2		C			M I
	;;
	shrp		x1 = w2, w1, 1		C			I0
	shr.u		x2 = w2, 1		C			I0
	br		.Lcj2			C			B

.grt2:	ld8		v3 = [vp], 8		C			M01
	ld8		u3 = [up], 8		C			M01
	;;
	ld8		v0 = [vp], 8		C			M01
	ld8		u0 = [up], 8		C			M01
	mov.i		ar.lc = n		C			I0
	;;
	ld8		v1 = [vp], 8		C			M01
	cmp.PRED	p9, p0 = w1, r10	C			M I
	ld8		u1 = [up], 8		C			M01
	and		r8 = 1, w1		C			M I
	;;
	ADDSUB		w2 = u2, v2		C			M I
	ld8		v2 = [vp], 8		C			M01
	;;
	cmp.PRED	p6, p0 = w2, u2		C			M I
	ld8		u2 = [up], 8		C			M01
	ADDSUB		w3 = u3, v3		C			M I
	br.cloop.dptk	.grt6			C			B
	;;

	cmp.PRED	p7, p0 = w3, u3		C			M I
   (p9)	cmp.eq.or	p6, p0 = LIM, w2	C			M I
   (p9)	add		w2 = INCR, w2		C			M I
	;;
	shrp		x1 = w2, w1, 1		C			I0
	ADDSUB		w0 = u0, v0		C			M I
   (p6)	cmp.eq.or	p7, p0 = LIM, w3	C			M I
   (p6)	add		w3 = INCR, w3		C			M I
	br		.Lcj6			C			B

.grt6:	ld8		v3 = [vp], 8		C			M01
	cmp.PRED	p7, p0 = w3, u3		C			M I
	ld8		u3 = [up], 8		C			M01
   (p9)	cmp.eq.or	p6, p0 = LIM, w2	C			M I
   (p9)	add		w2 = INCR, w2		C			M I
	;;
	shrp		x1 = w2, w1, 1		C			I0
	ADDSUB		w0 = u0, v0		C			M I
	ld8		v0 = [vp], 8		C			M01
   (p6)	cmp.eq.or	p7, p0 = LIM, w3	C			M I
   (p6)	add		w3 = INCR, w3		C			M I
	br		.LL10			C			B


.Lb11:	ld8		v1 = [vp], 8		C			M01
	ld8		u1 = [up], 8		C			M01
	shr.u		n = n, 2		C			I0
	;;
	ld8		v2 = [vp], 8		C			M01
	ld8		u2 = [up], 8		C			M01
	ADDSUB		w0 = r10, r11		C			M I
  (p15)	br.dpnt		.grt3			C			B
	;;

	cmp.PRED	p8, p0 = w0, r10	C			M I
	ADDSUB		w1 = u1, v1		C			M I
	and		r8 = 1, w0		C			M I
	;;
	cmp.PRED	p9, p0 = w1, u1		C			M I
	;;
	ADDSUB		w2 = u2, v2		C			M I
   (p8)	cmp.eq.or	p9, p0 = LIM, w1	C			M I
   (p8)	add		w1 = INCR, w1		C			M I
	;;
	cmp.PRED	p6, p0 = w2, u2		C			M I
	shrp		x0 = w1, w0, 1		C			I0
	;;
   (p9)	cmp.eq.or	p6, p0 = LIM, w2	C			M I
   (p9)	add		w2 = INCR, w2		C			M I
	br		.Lcj3			C			B

.grt3:	ld8		v3 = [vp], 8		C			M01
	ld8		u3 = [up], 8		C			M01
	;;
	ld8		v0 = [vp], 8		C			M01
	mov.i		ar.lc = n		C			I0
	cmp.PRED	p8, p0 = w0, r10	C			M I
	ld8		u0 = [up], 8		C			M01
	ADDSUB		w1 = u1, v1		C			M I
	and		r8 = 1, w0		C			M I
	;;
	ld8		v1 = [vp], 8		C			M01
	cmp.PRED	p9, p0 = w1, u1		C			M I
	ld8		u1 = [up], 8		C			M01
	;;
	ADDSUB		w2 = u2, v2		C			M I
	ld8		v2 = [vp], 8		C			M01
   (p8)	cmp.eq.or	p9, p0 = LIM, w1	C			M I
   (p8)	add		w1 = INCR, w1		C			M I
	;;
	cmp.PRED	p6, p0 = w2, u2		C			M I
	shrp		x0 = w1, w0, 1		C			I0
	ld8		u2 = [up], 8		C			M01
	ADDSUB		w3 = u3, v3		C			M I
	br.cloop.dptk	.grt7			C			B
	;;

	cmp.PRED	p7, p0 = w3, u3		C			M I
   (p9)	cmp.eq.or	p6, p0 = LIM, w2	C			M I
   (p9)	add		w2 = INCR, w2		C			M I
	br		.Lcj7			C			B

.grt7:	ld8		v3 = [vp], 8		C			M01
	cmp.PRED	p7, p0 = w3, u3		C			M I
	ld8		u3 = [up], 8		C			M01
   (p9)	cmp.eq.or	p6, p0 = LIM, w2	C			M I
   (p9)	add		w2 = INCR, w2		C			M I
	br		.LL11			C			B


C *** MAIN LOOP START ***
	ALIGN(32)
.Loop:	st8		[rp] = x3, 8		C			M23
	ld8		v3 = [vp], 8		C			M01
	cmp.PRED	p7, p0 = w3, u3		C			M I
	ld8		u3 = [up], 8		C			M01
   (p9)	cmp.eq.or	p6, p0 = LIM, w2	C			M I
   (p9)	add		w2 = INCR, w2		C			M I
	;;
.LL11:	st8		[rp] = x0, 8		C			M23
	shrp		x1 = w2, w1, 1		C			I0
	ADDSUB		w0 = u0, v0		C			M I
	ld8		v0 = [vp], 8		C			M01
   (p6)	cmp.eq.or	p7, p0 = LIM, w3	C			M I
   (p6)	add		w3 = INCR, w3		C			M I
	;;
.LL10:	cmp.PRED	p8, p0 = w0, u0		C			M I
	shrp		x2 = w3, w2, 1		C			I0
	nop.b		0
	ld8		u0 = [up], 8		C			M01
	ADDSUB		w1 = u1, v1		C			M I
	nop.b		0
	;;
	st8		[rp] = x1, 8		C			M23
	ld8		v1 = [vp], 8		C			M01
	cmp.PRED	p9, p0 = w1, u1		C			M I
	ld8		u1 = [up], 8		C			M01
   (p7)	cmp.eq.or	p8, p0 = LIM, w0	C			M I
   (p7)	add		w0 = INCR, w0		C			M I
	;;
.LL01:	st8		[rp] = x2, 8		C			M23
	shrp		x3 = w0, w3, 1		C			I0
	ADDSUB		w2 = u2, v2		C			M I
	ld8		v2 = [vp], 8		C			M01
   (p8)	cmp.eq.or	p9, p0 = LIM, w1	C			M I
   (p8)	add		w1 = INCR, w1		C			M I
	;;
.LL00:	cmp.PRED	p6, p0 = w2, u2		C			M I
	shrp		x0 = w1, w0, 1		C			I0
	nop.b		0
	ld8		u2 = [up], 8		C			M01
	ADDSUB		w3 = u3, v3		C			M I
	br.cloop.dptk	.Loop			C			B
	;;
C *** MAIN LOOP END ***

.Lskip:	st8		[rp] = x3, 8		C			M23
	cmp.PRED	p7, p0 = w3, u3		C			M I
   (p9)	cmp.eq.or	p6, p0 = LIM, w2	C			M I
   (p9)	add		w2 = INCR, w2		C			M I
	;;
.Lcj7:	st8		[rp] = x0, 8		C			M23
	shrp		x1 = w2, w1, 1		C			I0
	ADDSUB		w0 = u0, v0		C			M I
   (p6)	cmp.eq.or	p7, p0 = LIM, w3	C			M I
   (p6)	add		w3 = INCR, w3		C			M I
	;;
.Lcj6:	cmp.PRED	p8, p0 = w0, u0		C			M I
	shrp		x2 = w3, w2, 1		C			I0
	ADDSUB		w1 = u1, v1		C			M I
	;;
	st8		[rp] = x1, 8		C			M23
	cmp.PRED	p9, p0 = w1, u1		C			M I
   (p7)	cmp.eq.or	p8, p0 = LIM, w0	C			M I
   (p7)	add		w0 = INCR, w0		C			M I
	;;
.Lcj5:	st8		[rp] = x2, 8		C			M23
	shrp		x3 = w0, w3, 1		C			I0
	ADDSUB		w2 = u2, v2		C			M I
   (p8)	cmp.eq.or	p9, p0 = LIM, w1	C			M I
   (p8)	add		w1 = INCR, w1		C			M I
	;;
.Lcj4:	cmp.PRED	p6, p0 = w2, u2		C			M I
	shrp		x0 = w1, w0, 1		C			I0
	;;
	st8		[rp] = x3, 8		C			M23
   (p9)	cmp.eq.or	p6, p0 = LIM, w2	C			M I
   (p9)	add		w2 = INCR, w2		C			M I
	;;
.Lcj3:	st8		[rp] = x0, 8		C			M23
	shrp		x1 = w2, w1, 1		C			I0
	shr.u		x2 = w2, 1		C			I0
	;;
.Lcj2:	st8		[rp] = x1, 8		C			M23
   (p6)	dep		x2 = -1, x2, 63, 1	C			I0
	;;
.Lcj1:	st8		[rp] = x2		C			M23
	mov.i		ar.lc = r2		C			I0
	br.ret.sptk.many b0			C			B
EPILOGUE()
