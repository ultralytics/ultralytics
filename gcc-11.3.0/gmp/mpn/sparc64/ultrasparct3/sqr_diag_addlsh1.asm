dnl  SPARC v9 mpn_sqr_dial_addlsh1 for T3/T4/T5.

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

C		   cycles/limb
C UltraSPARC T3:	?
C UltraSPARC T4:	>= 4.5


define(`rp', `%i0')
define(`tp', `%i1')
define(`up', `%i2')
define(`n',  `%i3')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_sqr_diag_addlsh1)
	save	%sp, -176, %sp

	ldx	[up+0], %g1
	mulx	%g1, %g1, %o0
	umulxhi(%g1, %g1, %g2)
	stx	%o0, [rp+0]

	ldx	[up+8], %g1
	ldx	[tp+0], %g4
	ldx	[tp+8], %g5
	mulx	%g1, %g1, %o0
	orcc	%g0, %g0, %o5
	b	L(dm)
	 add	n, -2, n

	ALIGN(16)
L(top):	ldx	[up+8], %g1
	addcc	%g4, %o2, %o2
	addxccc(%g5, %o0, %g3)
	ldx	[tp+16], %g4
	ldx	[tp+24], %g5
	mulx	%g1, %g1, %o0
	stx	%o2, [rp+8]
	stx	%g3, [rp+16]
	add	rp, 16, rp
	add	tp, 16, tp
L(dm):	add	%g2, %o5, %o2
	umulxhi(%g1, %g1, %g2)
	addxccc(%g4, %g4, %g4)
	addxccc(%g5, %g5, %g5)
	add	up, 8, up
	addxc(	%g0, %g0, %o5)
	brnz	n, L(top)
	 add	n, -1, n

	addcc	%o2, %g4, %g4
	addxccc(%o0, %g5, %g5)
	stx	%g4, [rp+8]
	stx	%g5, [rp+16]
	addxc(	%o5, %g2, %g2)
	stx	%g2, [rp+24]

	ret
	 restore
EPILOGUE()
