dnl  IA-64 mpn_popcount -- mpn population count.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2000-2005 Free Software Foundation, Inc.

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
C Itanium:       1.5
C Itanium 2:     1

C INPUT PARAMETERS
define(`up', `r32')
define(`n', `r33')

define(`u0',`r16') define(`u1',`r17') define(`u2',`r18') define(`u3',`r19')
define(`c0',`r28') define(`c1',`r29') define(`c2',`r30') define(`c3',`r31')
define(`s',`r8')


ASM_START()
PROLOGUE(mpn_popcount)
	.prologue
ifdef(`HAVE_ABI_32',
`	addp4		up = 0, up		C			M I
	nop.m		0
	zxt4		n = n			C			I
	;;
')

 {.mmi;	add		r9 = 512, up		C prefetch pointer	M I
	ld8		r10 = [up], 8		C load first limb	M01
	mov.i		r2 = ar.lc		C save ar.lc		I0
}{.mmi;	and		r14 = 3, n		C			M I
	cmp.lt		p15, p14 = 4, n		C small count?		M I
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
	shr.u		n = n, 2		C			I0
	mov		s = 0			C			M I
	;;
	ld8		u2 = [up], 8		C			M01
	popcnt		c0 = r10		C			I0
	mov.i		ar.lc = n		C			I0
	;;
	ld8		u3 = [up], 8		C			M01
	popcnt		c1 = u1			C			I0
  (p15)	br.cond.dptk	.grt4			C			B
	;;
	nop.m	0				C			-
	nop.m	0				C			-
	popcnt		c2 = u2			C			I0
	;;
	mov		s = c0			C			M I
	popcnt		c3 = u3			C			I0
	br		.Lcj4			C			B

.grt4:	ld8		u0 = [up], 8		C			M01
	popcnt		c2 = u2			C			I0
	br		.LL00			C			B


.Lb01:
	popcnt		s = r10			C			I0
  (p14)	br.ret.sptk.many b0			C			B

.grt1:	ld8		u0 = [up], 8		C			M01
	shr.u		n = n, 2		C			I0
	;;
	ld8		u1 = [up], 8		C			M01
	mov.i		ar.lc = n		C			I0
	;;
	ld8		u2 = [up], 8		C			M01
	popcnt		c0 = u0			C			I0
	mov		c3 = 0			C			I0

	;;
	ld8		u3 = [up], 8		C			M01
	popcnt		c1 = u1			C			I0
	br.cloop.dptk	.Loop			C			B
	br		.Lend			C			B


.Lb10:	ld8		u3 = [up], 8		C			M01
	shr.u		n = n, 2		C			I0
  (p15)	br.cond.dptk	.grt2			C			B

	popcnt		s = r10			C			I0
	;;
	popcnt		c3 = u3			C			I0
	br		.Lcj2			C			B

.grt2:	ld8		u0 = [up], 8		C			M01
	mov.i		ar.lc = n		C			I0
	popcnt		c2 = r10		C			I0
	;;
	ld8		u1 = [up], 8		C			M01
	popcnt		c3 = u3			C			I0
	mov		s = 0			C			M I
	;;
	ld8		u2 = [up], 8		C			M01
	popcnt		c0 = u0			C			I0
	br		.LL10			C			B


.Lb11:	ld8		u2 = [up], 8		C			M01
	shr.u		n = n, 2		C			I0
	mov		s = 0			C			M I
	;;
	ld8		u3 = [up], 8		C			M01
	popcnt		s = r10			C			I0
  (p15)	br.cond.dptk	.grt3			C			B

	popcnt		c2 = u2			C			I0
	;;
	popcnt		c3 = u3			C			I0
	br		.Lcj3			C			B

.grt3:	ld8		u0 = [up], 8		C			M01
	popcnt		c2 = u2			C			I0
	mov.i		ar.lc = n		C			I0
	mov		c1 = 0
	;;
	ld8		u1 = [up], 8		C			M01
	popcnt		c3 = u3			C			I0
	br		.LL11			C			B


.Loop:	ld8		u0 = [up], 8		C			M01
	popcnt		c2 = u2			C			I0
	add		s = s, c3		C			M I
	;;
.LL00:	ld8		u1 = [up], 8		C			M01
	popcnt		c3 = u3			C			I0
	add		s = s, c0		C			M I
	;;
.LL11:	ld8		u2 = [up], 8		C			M01
	popcnt		c0 = u0			C			I0
	add		s = s, c1		C			M I
	;;
.LL10:	ld8		u3 = [up], 8		C			M01
	popcnt		c1 = u1			C			I0
	add		s = s, c2		C			M I
	lfetch		[r9], 32		C			M01
	nop.m		0			C			-
	br.cloop.dptk	.Loop			C			B
	;;

.Lend:	popcnt		c2 = u2			C			I0
	add		s = s, c3		C			M I
	;;
	popcnt		c3 = u3			C			I0
	add		s = s, c0		C			M I
	;;
.Lcj4:	add		s = s, c1		C			M I
	;;
.Lcj3:	add		s = s, c2		C			M I
	;;
.Lcj2:	add		s = s, c3		C			M I
	mov.i		ar.lc = r2		C			I0
	br.ret.sptk.many b0			C			B
EPILOGUE()
ASM_END()
