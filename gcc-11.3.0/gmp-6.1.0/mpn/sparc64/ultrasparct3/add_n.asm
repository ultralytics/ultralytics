dnl  SPARC v9 mpn_add_n for T3/T4.

dnl  Contributed to the GNU project by David Miller.

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
C UltraSPARC T3:	 8
C UltraSPARC T4:	 3

C INPUT PARAMETERS
define(`rp', `%i0')
define(`up', `%i1')
define(`vp', `%i2')
define(`n',  `%i3')
define(`cy', `%i4')

define(`u0_off', `%l2')
define(`u1_off', `%l3')
define(`loop_n', `%l6')
define(`tmp', `%l7')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_add_nc)
	save	%sp, -176, %sp
	b,a	L(ent)
EPILOGUE()
PROLOGUE(mpn_add_n)
	save	%sp, -176, %sp

	mov	0, cy
L(ent):
	subcc	n, 1, n
	be	L(final_one)
	 cmp	%g0, cy

	ldx	[up + 0], %o4
	sllx	n, 3, tmp

	ldx	[vp + 0], %o5
	add	up, tmp, u0_off

	ldx	[up + 8], %g5
	neg	tmp, loop_n

	ldx	[vp + 8], %g1
	add	u0_off, 8, u1_off

	sub	loop_n, -(2 * 8), loop_n

	brgez,pn loop_n, L(loop_tail)
	 add	vp, (2 * 8), vp

	b,a	L(top)
	ALIGN(16)
L(top):
	addxccc(%o4, %o5, tmp)
	ldx	[vp + 0], %o5

	add	rp, (2 * 8), rp
	ldx	[loop_n + u0_off], %o4

	add	vp, (2 * 8), vp
	stx	tmp, [rp - 16]

	addxccc(%g1, %g5, tmp)
	ldx	[vp - 8], %g1

	ldx	[loop_n + u1_off], %g5
	sub	loop_n, -(2 * 8), loop_n

	brlz	loop_n, L(top)
	 stx	tmp, [rp - 8]

L(loop_tail):
	addxccc(%o4, %o5, %g3)
	add	loop_n, u0_off, up

	addxccc(%g1, %g5, %g5)
	stx	%g3, [rp + 0]

	brgz,pt	loop_n, L(done)
	 stx	%g5, [rp + 8]

	add	rp, (2 * 8), rp
L(final_one):
	ldx	[up+0], %o4
	ldx	[vp+0], %o5
	addxccc(%o4, %o5, %g3)
	stx	%g3, [rp+0]

L(done):
	addxc(%g0, %g0, %i0)
	ret
	 restore
EPILOGUE()
