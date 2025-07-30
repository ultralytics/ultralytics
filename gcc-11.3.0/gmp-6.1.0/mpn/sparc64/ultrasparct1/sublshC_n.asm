dnl  SPARC v9 mpn_sublshC_n for T1/T2.

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

C		   cycles/limb
C UltraSPARC T1:	21
C UltraSPARC T2:	 ?

C INPUT PARAMETERS
define(`rp', `%o0')
define(`up', `%o1')
define(`vp', `%o2')
define(`n',  `%o3')
define(`cy', `%o4')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(func)
	mov	0, cy
	mov	0, %g5
	cmp	%g0, cy
L(top):	ldx	[up+0], %o4
	add	up, 8, up
	ldx	[vp+0], %o5
	add	vp, 8, vp
	add	rp, 8, rp

	sllx	%o5, LSH, %g4
	add	n, -1, n
	or	%g5, %g4, %g4
	srlx	%o5, RSH, %g5

	srlx	%o4, 32, %g1
	srlx	%g4, 32, %g2
	subccc	%o4, %g4, %g3
	subccc	%g1, %g2, %g0
	brgz	n, L(top)
	 stx	%g3, [rp-8]

	retl
	addc	%g5, %g0, %o0
EPILOGUE()
