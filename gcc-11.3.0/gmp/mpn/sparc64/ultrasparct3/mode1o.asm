dnl  SPARC T3/T4/T5 mpn_modexact_1c_odd.

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

C                  cycles/limb
C UltraSPARC T3:	30
C UltraSPARC T4/T5:	26

C INPUT PARAMETERS
define(`ap',  `%o0')
define(`n',   `%o1')
define(`d',   `%o2')
define(`cy',  `%o3')

define(`dinv',`%o5')
define(`a0',  `%g1')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_modexact_1c_odd)
	srlx	d, 1, %g1
	and	%g1, 127, %g1

	LEA64(binvert_limb_table, g2, g4)
	ldub	[%g2+%g1], %g1
	add	%g1, %g1, %g2
	mulx	%g1, %g1, %g1
	mulx	%g1, d, %g1
	sub	%g2, %g1, %g2
	add	%g2, %g2, %g1
	mulx	%g2, %g2, %g2
	mulx	%g2, d, %g2
	sub	%g1, %g2, %g1
	add	%g1, %g1, %o5
	mulx	%g1, %g1, %g1
	mulx	%g1, d, %g1
	sub	%o5, %g1, dinv
	add	n, -1, n

L(top):	ldx	[ap], a0
	add	ap, 8, ap
	subcc	a0, cy, %g3
	mulx	%g3, dinv, %g5
	umulxhi(d, %g5, %g5)
	addxc(	%g5, %g0, cy)
	brnz,pt	n, L(top)
	 add	n, -1, n

	retl
	 mov	cy, %o0
EPILOGUE()
