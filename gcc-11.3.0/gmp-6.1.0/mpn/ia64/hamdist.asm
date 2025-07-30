dnl  IA-64 mpn_hamdist -- mpn hamming distance.

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

C           cycles/limb
C Itanium:       2
C Itanium 2:     1

C INPUT PARAMETERS
define(`up', `r32')
define(`vp', `r33')
define(`n', `r34')

define(`u0',`r16') define(`u1',`r17') define(`u2',`r18') define(`u3',`r19')
define(`v0',`r20') define(`v1',`r21') define(`v2',`r22') define(`v3',`r23')
define(`x0',`r24') define(`x1',`r25') define(`x2',`r26') define(`x3',`r27')
define(`c0',`r28') define(`c1',`r29') define(`c2',`r30') define(`c3',`r31')
define(`s',`r8')


ASM_START()
PROLOGUE(mpn_hamdist)
	.prologue
ifdef(`HAVE_ABI_32',
`	addp4		up = 0, up		C			M I
	addp4		vp = 0, vp		C			M I
	zxt4		n = n			C			I
	;;
')

 {.mmi;	ld8		r10 = [up], 8		C load first ulimb	M01
	ld8		r11 = [vp], 8		C load first vlimb	M01
	mov.i		r2 = ar.lc		C save ar.lc		I0
}{.mmi;	and		r14 = 3, n		C			M I
	cmp.lt		p15, p0 = 4, n		C small count?		M I
	add		n = -5, n		C			M I
	;;
}{.mmi;	cmp.eq		p6, p0 = 1, r14		C			M I
	cmp.eq		p7, p0 = 2, r14		C			M I
	cmp.eq		p8, p0 = 3, r14		C			M I
}{.bbb
  (p6)	br.dptk		.Lb01			C			B
  (p7)	br.dptk		.Lb10			C			B
  (p8)	br.dptk		.Lb11			C			B
}


.Lb00:	ld8		u1 = [up], 8		C			M01
	ld8		v1 = [vp], 8		C			M01
	shr.u		n = n, 2		C			I0
	xor		x0 = r10, r11		C			M I
	;;
	ld8		u2 = [up], 8		C			M01
	ld8		v2 = [vp], 8		C			M01
	mov.i		ar.lc = n		C			I0
	xor		x1 = u1, v1		C			M I
	;;
	ld8		u3 = [up], 8		C			M01
	ld8		v3 = [vp], 8		C			M01
	xor		x2 = u2, v2		C			M I
	mov		s = 0			C			M I
  (p15)	br.cond.dptk	.grt4			C			B
	;;
	popcnt		c0 = x0			C			I0
	xor		x3 = u3, v3		C			M I
	;;
	popcnt		c1 = x1			C			I0
	;;
	popcnt		c2 = x2			C			I0
	br		.Lcj4			C			B

.grt4:	ld8		u0 = [up], 8		C			M01
	ld8		v0 = [vp], 8		C			M01
	xor		x1 = u1, v1		C			M I
	;;
	ld8		u1 = [up], 8		C			M01
	ld8		v1 = [vp], 8		C			M01
	xor		x2 = u2, v2		C			M I
	;;
	ld8		u2 = [up], 8		C			M01
	ld8		v2 = [vp], 8		C			M01
	popcnt		c0 = x0			C			I0
	xor		x3 = u3, v3		C			M I
	;;
	ld8		u3 = [up], 8		C			M01
	ld8		v3 = [vp], 8		C			M01
	popcnt		c1 = x1			C			I0
	xor		x0 = u0, v0		C			M I
	br.cloop.dpnt	.grt8			C			B

	popcnt		c2 = x2			C			I0
	xor		x1 = u1, v1		C			M I
	br		.Lcj8			C			B

.grt8:	ld8		u0 = [up], 8		C			M01
	ld8		v0 = [vp], 8		C			M01
	popcnt		c2 = x2			C			I0
	xor		x1 = u1, v1		C			M I
	br		.LL00			C			B


.Lb01:	xor		x3 = r10, r11		C			M I
	shr.u		n = n, 2		C			I0
  (p15)	br.cond.dptk	.grt1			C			B
	;;
	popcnt		r8 = x3			C			I0
	br.ret.sptk.many b0			C			B

.grt1:	ld8		u0 = [up], 8		C			M01
	ld8		v0 = [vp], 8		C			M01
	mov.i		ar.lc = n		C			I0
	;;
	ld8		u1 = [up], 8		C			M01
	ld8		v1 = [vp], 8		C			M01
	mov		s = 0			C			M I
	;;
	ld8		u2 = [up], 8		C			M01
	ld8		v2 = [vp], 8		C			M01
	;;
	ld8		u3 = [up], 8		C			M01
	ld8		v3 = [vp], 8		C			M01
	xor		x0 = u0, v0		C			M I
	br.cloop.dpnt	.grt5			C			B

	xor		x1 = u1, v1		C			M I
	;;
	popcnt		c3 = x3			C			I0
	xor		x2 = u2, v2		C			M I
	;;
	popcnt		c0 = x0			C			I0
	xor		x3 = u3, v3		C			M I
	;;
	popcnt		c1 = x1			C			I0
	br		.Lcj5			C			B

.grt5:	ld8		u0 = [up], 8		C			M01
	ld8		v0 = [vp], 8		C			M01
	xor		x1 = u1, v1		C			M I
	;;
	ld8		u1 = [up], 8		C			M01
	ld8		v1 = [vp], 8		C			M01
	popcnt		c3 = x3			C			I0
	xor		x2 = u2, v2		C			M I
	;;
	ld8		u2 = [up], 8		C			M01
	ld8		v2 = [vp], 8		C			M01
	popcnt		c0 = x0			C			I0
	xor		x3 = u3, v3		C			M I
	;;
	ld8		u3 = [up], 8		C			M01
	ld8		v3 = [vp], 8		C			M01
	popcnt		c1 = x1			C			I0
	xor		x0 = u0, v0		C			M I
	br.cloop.dpnt	.Loop			C			B
	br		.Lend			C			B


.Lb10:	ld8		u3 = [up], 8		C			M01
	ld8		v3 = [vp], 8		C			M01
	xor		x2 = r10, r11		C			M I
  (p15)	br.cond.dptk	.grt2			C			B
	;;
	xor		x3 = u3, v3		C			M I
	;;
	popcnt		c2 = x2			C			I0
	;;
	popcnt		c3 = x3			C			I0
	;;
	add		s = c2, c3		C			M I
	br.ret.sptk.many b0			C			B

.grt2:	ld8		u0 = [up], 8		C			M01
	ld8		v0 = [vp], 8		C			M01
	shr.u		n = n, 2		C			I0
	;;
	ld8		u1 = [up], 8		C			M01
	ld8		v1 = [vp], 8		C			M01
	mov.i		ar.lc = n		C			I0
	mov		s = 0			C			M I
	;;
	ld8		u2 = [up], 8		C			M01
	ld8		v2 = [vp], 8		C			M01
	xor		x3 = u3, v3		C			M I
	;;
	ld8		u3 = [up], 8		C			M01
	ld8		v3 = [vp], 8		C			M01
	xor		x0 = u0, v0		C			M I
	br.cloop.dptk	.grt6			C			B

	popcnt		c2 = x2			C			I0
	xor		x1 = u1, v1		C			M I
	;;
	popcnt		c3 = x3			C			I0
	xor		x2 = u2, v2		C			M I
	;;
	popcnt		c0 = x0			C			I0
	xor		x3 = u3, v3		C			M I
	br		.Lcj6			C			B

.grt6:	ld8		u0 = [up], 8		C			M01
	ld8		v0 = [vp], 8		C			M01
	popcnt		c2 = x2			C			I0
	xor		x1 = u1, v1		C			M I
	;;
	ld8		u1 = [up], 8		C			M01
	ld8		v1 = [vp], 8		C			M01
	popcnt		c3 = x3			C			I0
	xor		x2 = u2, v2		C			M I
	;;
	ld8		u2 = [up], 8		C			M01
	ld8		v2 = [vp], 8		C			M01
	popcnt		c0 = x0			C			I0
	xor		x3 = u3, v3		C			M I
	br		.LL10			C			B


.Lb11:	ld8		u2 = [up], 8		C			M01
	ld8		v2 = [vp], 8		C			M01
	shr.u		n = n, 2		C			I0
	xor		x1 = r10, r11		C			M I
	;;
	ld8		u3 = [up], 8		C			M01
	ld8		v3 = [vp], 8		C			M01
	xor		x2 = u2, v2		C			M I
  (p15)	br.cond.dptk	.grt3			C			B
	;;
	xor		x3 = u3, v3		C			M I
	;;
	popcnt		c1 = x1			C			I0
	;;
	popcnt		c2 = x2			C			I0
	;;
	popcnt		c3 = x3			C			I0
	;;
	add		s = c1, c2		C			M I
	;;
	add		s = s, c3		C			M I
	br.ret.sptk.many b0			C			B

.grt3:	ld8		u0 = [up], 8		C			M01
	ld8		v0 = [vp], 8		C			M01
	mov.i		ar.lc = n		C			I0
	;;
	ld8		u1 = [up], 8		C			M01
	ld8		v1 = [vp], 8		C			M01
	mov		s = 0			C			M I
	;;
	ld8		u2 = [up], 8		C			M01
	ld8		v2 = [vp], 8		C			M01
	xor		x3 = u3, v3		C			M I
	;;
	ld8		u3 = [up], 8		C			M01
	ld8		v3 = [vp], 8		C			M01
	popcnt		c1 = x1			C			I0
	xor		x0 = u0, v0		C			M I
	br.cloop.dptk	.grt7			C			B
	popcnt		c2 = x2			C			I0
	xor		x1 = u1, v1		C			M I
	;;
	popcnt		c3 = x3			C			I0
	xor		x2 = u2, v2		C			M I
	br		.Lcj7			C			B

.grt7:	ld8		u0 = [up], 8		C			M01
	ld8		v0 = [vp], 8		C			M01
	popcnt		c2 = x2			C			I0
	xor		x1 = u1, v1		C			M I
	;;
	ld8		u1 = [up], 8		C			M01
	ld8		v1 = [vp], 8		C			M01
	popcnt		c3 = x3			C			I0
	xor		x2 = u2, v2		C			M I
	br		.LL11			C			B


	ALIGN(32)
.Loop:	ld8		u0 = [up], 8		C			M01
	ld8		v0 = [vp], 8		C			M01
	popcnt		c2 = x2			C			I0
	add		s = s, c3		C			M I
	xor		x1 = u1, v1		C			M I
	nop.b		1			C			-
	;;
.LL00:	ld8		u1 = [up], 8		C			M01
	ld8		v1 = [vp], 8		C			M01
	popcnt		c3 = x3			C			I0
	add		s = s, c0		C			M I
	xor		x2 = u2, v2		C			M I
	nop.b		1			C			-
	;;
.LL11:	ld8		u2 = [up], 8		C			M01
	ld8		v2 = [vp], 8		C			M01
	popcnt		c0 = x0			C			I0
	add		s = s, c1		C			M I
	xor		x3 = u3, v3		C			M I
	nop.b		1			C			-
	;;
.LL10:	ld8		u3 = [up], 8		C			M01
	ld8		v3 = [vp], 8		C			M01
	popcnt		c1 = x1			C			I0
	add		s = s, c2		C			M I
	xor		x0 = u0, v0		C			M I
	br.cloop.dptk	.Loop			C			B
	;;

.Lend:	popcnt		c2 = x2			C			I0
	add		s = s, c3		C			M I
	xor		x1 = u1, v1		C			M I
	;;
.Lcj8:	popcnt		c3 = x3			C			I0
	add		s = s, c0		C			M I
	xor		x2 = u2, v2		C			M I
	;;
.Lcj7:	popcnt		c0 = x0			C			I0
	add		s = s, c1		C			M I
	xor		x3 = u3, v3		C			M I
	;;
.Lcj6:	popcnt		c1 = x1			C			I0
	add		s = s, c2		C			M I
	;;
.Lcj5:	popcnt		c2 = x2			C			I0
	add		s = s, c3		C			M I
	;;
.Lcj4:	popcnt		c3 = x3			C			I0
	add		s = s, c0		C			M I
	;;
	add		s = s, c1		C			M I
	;;
	add		s = s, c2		C			M I
	;;
	add		s = s, c3		C			M I
	mov.i		ar.lc = r2		C			I0
	br.ret.sptk.many b0			C			B
EPILOGUE()
ASM_END()
