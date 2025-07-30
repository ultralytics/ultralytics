dnl  SPARC T1 32-bit mpn_mul_1.

dnl  Contributed to the GNU project by David Miller.

dnl  Copyright 2010, 2013 Free Software Foundation, Inc.

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
C UltraSPARC T1:       20
C UltraSPARC T2:       18
C UltraSPARC T3:       18
C UltraSPARC T4:       4

C INPUT PARAMETERS
define(`rp',	`%o0')
define(`up',	`%o1')
define(`n',	`%o2')
define(`v0',	`%o3')

ASM_START()
PROLOGUE(mpn_mul_1)
	srl	n, 0, n
	srl	v0, 0, v0
	subcc	n, 1, n
	be	L(final_one)
	 clr	%o5

L(top):	lduw	[up+0], %g1
	lduw	[up+4], %g2
	mulx	%g1, v0, %g3
	add	up, 8, up
	mulx	%g2, v0, %o4
	sub	n, 2, n
	add	rp, 8, rp
	add	%o5, %g3, %g3
	stw	%g3, [rp-8]
	srlx	%g3, 32, %o5
	add	%o5, %o4, %o4
	stw	%o4, [rp-4]
	brgz	n, L(top)
	 srlx	%o4, 32, %o5

	brlz,pt	n, L(done)
	 nop

L(final_one):
	lduw	[up+0], %g1
	mulx	%g1, v0, %g3
	add	%o5, %g3, %g3
	stw	%g3, [rp+0]
	srlx	%g3, 32, %o5

L(done):
	retl
	 mov	%o5, %o0
EPILOGUE()
