dnl  SPARC v9 mpn_sub_n for T3/T4.

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

define(`u0_off', `%l0')
define(`u1_off', `%l1')
define(`v0_off', `%l2')
define(`v1_off', `%l3')
define(`r0_off', `%l4')
define(`r1_off', `%l5')
define(`loop_n', `%l6')
define(`tmp', `%l7')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_sub_nc)
	save	%sp, -176, %sp
	ba,pt	%xcc, L(ent)
	 xor	cy, 1, cy
EPILOGUE()
PROLOGUE(mpn_sub_n)
	save	%sp, -176, %sp
	mov	1, cy
L(ent):
	subcc	n, 1, n
	be	L(final_one)
	 cmp	%g0, cy

	ldx	[up + 0], %o4
	sllx	n, 3, tmp

	ldx	[vp + 0], %o5
	add	up, tmp, u0_off

	ldx	[up + 8], %g5
	add	vp, tmp, v0_off

	ldx	[vp + 8], %g1
	add	rp, tmp, r0_off

	neg	tmp, loop_n
	add	u0_off, 8, u1_off

	add	v0_off, 8, v1_off
	sub	loop_n, -(2 * 8), loop_n

	sub	r0_off, 16, r0_off
	brgez,pn loop_n, L(loop_tail)
	 sub	r0_off, 8, r1_off

	b,a	L(top)
	ALIGN(16)
L(top):
	xnor	%o5, 0, tmp
	ldx	[loop_n + v0_off], %o5

	addxccc(%o4, tmp, %g3)
	ldx	[loop_n + u0_off], %o4

	xnor	%g1, 0, %g1
	stx	%g3, [loop_n + r0_off]

	addxccc(%g5, %g1, tmp)
	ldx	[loop_n + v1_off], %g1

	ldx	[loop_n + u1_off], %g5
	sub	loop_n, -(2 * 8), loop_n

	brlz	loop_n, L(top)
	 stx	tmp, [loop_n + r1_off]

L(loop_tail):
	xnor	%o5, 0, tmp
	xnor	%g1, 0, %g1

	addxccc(%o4, tmp, %g3)
	add	loop_n, u0_off, up

	addxccc(%g5, %g1, %g5)
	add	loop_n, r0_off, rp

	stx	%g3, [rp + 0]
	add	loop_n, v0_off, vp

	brgz,pt	loop_n, L(done)
	 stx	%g5, [rp + 8]

	add	rp, (2 * 8), rp

L(final_one):
	ldx	[up+0], %o4
	ldx	[vp+0], %o5
	xnor	%o5, %g0, %o5
	addxccc(%o4, %o5, %g3)
	stx	%g3, [rp+0]

L(done):
	clr	%i0
	movcc	%xcc, 1, %i0
	ret
	 restore
EPILOGUE()
