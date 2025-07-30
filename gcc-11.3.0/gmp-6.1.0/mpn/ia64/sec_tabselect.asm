dnl  IA-64 mpn_sec_tabselect.

dnl  Copyright 2011 Free Software Foundation, Inc.

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
C Itanium:       ?
C Itanium 2:     2.5

C NOTES
C  * Using software pipelining could trivially yield 2 c/l without unrolling,
C    or 1+epsilon with unrolling.  (This code was modelled after the powerpc64
C    code, for simplicity.)

C mpn_sec_tabselect (mp_limb_t *rp, mp_limb_t *tp, mp_size_t n, mp_size_t nents, mp_size_t which)
define(`rp',     `r32')
define(`tp',     `r33')
define(`n',      `r34')
define(`nents',  `r35')
define(`which',  `r36')

define(`mask',   `r8')

define(`rp1',     `r32')
define(`tp1',     `r33')
define(`rp2',     `r14')
define(`tp2',     `r15')

ASM_START()
PROLOGUE(mpn_sec_tabselect)
	.prologue
	.save	ar.lc, r2
	.body
ifdef(`HAVE_ABI_32',`
 {.mmi;	addp4	rp = 0, rp		C			M I
	addp4	tp = 0, tp		C			M I
	zxt4	n = n			C			I
}{.mii;	nop	0
	zxt4	nents = nents		C			I
	zxt4	which = which		C			I
	;;
}')
 {.mmi;	add	rp2 = 8, rp1
	add	tp2 = 8, tp1
	add	r6 = -2, n
	;;
}{.mmi;	cmp.eq	p10, p0 = 1, n
	and	r9 = 1, n		C set cr0 for use in inner loop
	shr.u	r6 = r6, 1		C inner loop count
	;;
}{.mmi;	cmp.eq	p8, p0 = 0, r9
	sub	which = nents, which
	shl	n = n, 3
	;;
}
L(outer):
 {.mmi;	cmp.eq	p6, p7 = which, nents	C are we at the selected table entry?
	nop	0
	mov	ar.lc = r6		C			I0
	;;
}{.mmb;
  (p6)	mov	mask = -1
  (p7)	mov	mask = 0
  (p8)	br.dptk	L(top)			C branch to loop entry if n even
	;;
}{.mmi;	ld8	r16 = [tp1], 8
	add	tp2 = 8, tp2
	nop	0
	;;
}{.mmi;	ld8	r18 = [rp1]
	and	r16 = r16, mask
	nop	0
	;;
}{.mmi;	andcm	r18 = r18, mask
	;;
	or	r16 = r16, r18
	nop	0
	;;
}{.mmb;	st8	[rp1] = r16, 8
	add	rp2 = 8, rp2
  (p10)	br.dpnt	L(end)
}
	ALIGN(32)
L(top):
 {.mmi;	ld8	r16 = [tp1], 16
	ld8	r17 = [tp2], 16
	nop	0
	;;
}{.mmi;	ld8	r18 = [rp1]
	and	r16 = r16, mask
	nop	0
}{.mmi;	ld8	r19 = [rp2]
	and	r17 = r17, mask
	nop	0
	;;
}{.mmi;	andcm	r18 = r18, mask
	andcm	r19 = r19, mask
	nop	0
	;;
}{.mmi;	or	r16 = r16, r18
	or	r17 = r17, r19
	nop	0
	;;
}{.mmb;	st8	[rp1] = r16, 16
	st8	[rp2] = r17, 16
	br.cloop.dptk	L(top)
	;;
}
L(end):
 {.mmi;	sub	rp1 = rp1, n		C move rp back to beginning
	sub	rp2 = rp2, n		C move rp back to beginning
	cmp.ne	p9, p0 = 1, nents
}{.mmb;	add	nents = -1, nents
	nop	0
  (p9)	br.dptk	L(outer)
	;;
}{.mib;	nop	0
	nop	0
	br.ret.sptk.many b0
}
EPILOGUE()
