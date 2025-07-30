dnl  SPARC v9 mpn_rshift

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

C		    cycles/limb
C UltraSPARC 1&2:	 2
C UltraSPARC 3:		 2.5
C UltraSPARC T1:	17.5
C UltraSPARC T3:	 8
C UltraSPARC T4:	 3

C INPUT PARAMETERS
define(`rp',     `%i0')
define(`up',     `%i1')
define(`n',      `%i2')
define(`cnt',    `%i3')

define(`tcnt',   `%i4')
define(`retval', `%i5')
define(`u0',     `%l0')
define(`u1',     `%l1')
define(`r0',     `%l6')
define(`r1',     `%l7')
define(`u0_off', `%o0')
define(`u1_off', `%o1')
define(`r0_off', `%o2')
define(`r1_off', `%o3')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_rshift)
	save	%sp, -176, %sp

	sllx	n, 3, n
	sub	%g0, cnt, tcnt

	add	up, n, up
	add	rp, n, rp

	neg	n, n
	sub	up, (2 * 8), u0_off
	sub	rp, (5 * 8), r0_off

	ldx	[n + up], u1		C WAS: up + 0
	sub	u0_off, (1 * 8), u1_off
	sub	r0_off, (1 * 8), r1_off

	subcc	n, -(3 * 8), n
	sllx	u1, tcnt, retval

	bg,pn	%xcc, L(end12)
	 srlx	u1, cnt, %l3

	ldx	[n + u0_off], u0	C WAS: up + 0
	subcc	n, -(2 * 8), n

	ldx	[n + u1_off], u1	C WAS: up + 8

	bg,pn	%xcc, L(end34)
	 sllx	u0, tcnt, %l4

	b,a	L(top)
	ALIGN(16)
L(top):
	srlx	u0, cnt, %l2
	or	%l3, %l4, r0

	ldx	[n + u0_off], u0	C WAS: up + 0
	sllx	u1, tcnt, %l5

	stx	r0, [n + r0_off]	C WAS: rp + 0
	subcc	n, -(2 * 8), n

	srlx	u1, cnt, %l3
	or	%l2, %l5, r1

	ldx	[n + u1_off], u1	C WAS: up + 8
	sllx	u0, tcnt, %l4

	ble,pt	%xcc, L(top)
	 stx	r1, [n + r1_off]	C WAS: rp + 8

L(end34):
	srlx	u0, cnt, %l2
	or	%l3, %l4, r0

	sllx	u1, tcnt, %l5
	stx	r0, [n + r0_off]	C WAS: rp + 0

	or	%l2, %l5, r1
	sub	n, -(2 * 8), %o5

	srlx	u1, cnt, %l3
	stx	r1, [%o5 + r1_off]	C WAS: rp + 8

L(end12):
	andcc	n, 8, %g0
	bz,pn	%xcc, L(done)
	 nop

	ldx	[n + u0_off], u1
	sllx	u1, tcnt, %l4
	or	%l3, %l4, r0
	stx	r0, [r0_off + 24]
	srlx	u1, cnt, %l3
L(done):
	stx	%l3, [r0_off + 32]

	ret
	restore retval, 0, %o0
EPILOGUE()
