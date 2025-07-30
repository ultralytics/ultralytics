dnl  SPARC T1 32-bit mpn_submul_1.

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
C UltraSPARC T1:       24
C UltraSPARC T2:       19
C UltraSPARC T3:       19
C UltraSPARC T4:       5

C INPUT PARAMETERS
define(`rp',	`%i0')
define(`up',	`%i1')
define(`n',	`%i2')
define(`v0',	`%i3')

ASM_START()
PROLOGUE(mpn_submul_1)
	save	%sp, -96, %sp
	srl	n, 0, %o4
	srl	v0, 0, %g1
	subcc	%o4, 1, %o4
	be	L(final_one)
	 subcc	%g0, 0, %o5

L(top):	lduw	[up+0], %l0
	lduw	[rp+0], %l2
	lduw	[up+4], %l1
	lduw	[rp+4], %l3
	mulx	%l0, %g1, %g3
	add	up, 8, up
	mulx	%l1, %g1, %o3
	sub	%o4, 2, %o4
	add	rp, 8, rp
	addx	%o5, %g3, %g3
	srlx	%g3, 32, %o5
	subcc	%l2, %g3, %g3
	stw	%g3, [rp-8]
	addx	%o5, %o3, %o3
	srlx	%o3, 32, %o5
	subcc	%l3, %o3, %o3
	brgz	%o4, L(top)
	 stw	%o3, [rp-4]

	brlz,pt	%o4, L(done)
	 nop

L(final_one):
	lduw	[up+0], %l0
	lduw	[rp+0], %l2
	mulx	%l0, %g1, %g3
	addx	%o5, %g3, %g3
	srlx	%g3, 32, %o5
	subcc	%l2, %g3, %g3
	stw	%g3, [rp+0]

L(done):
	addx	%o5, 0, %o5
	ret
	 restore %o5, 0, %o0
EPILOGUE()
