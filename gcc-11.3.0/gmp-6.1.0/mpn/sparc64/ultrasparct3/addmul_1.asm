dnl  SPARC v9 mpn_addmul_1 for T3/T4/T5.

dnl  Contributed to the GNU project by David Miller and Torbjörn Granlund.

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
C UltraSPARC T3:	26
C UltraSPARC T4:	4.5

C INPUT PARAMETERS
define(`rp', `%i0')
define(`up', `%i1')
define(`n',  `%i2')
define(`v0', `%i3')

define(`u0',  `%l0')
define(`u1',  `%l1')
define(`u2',  `%l2')
define(`u3',  `%l3')
define(`r0',  `%l4')
define(`r1',  `%l5')
define(`r2',  `%l6')
define(`r3',  `%l7')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_addmul_1)
	save	%sp, -176, %sp
	ldx	[up+0], %g1

	and	n, 3, %g3
	brz	%g3, L(b0)
	 addcc	%g0, %g0, %g5			C clear carry limb, flag
	cmp	%g3, 2
	bcs	%xcc, L(b01)
	 nop
	be	%xcc, L(b10)
	 ldx	[up+8], %g5

L(b11):	ldx	[up+16], u3
	mulx	%g1, v0, %o2
	umulxhi(%g1, v0, %o3)
	ldx	[rp+0], r1
	mulx	%g5, v0, %o4
	ldx	[rp+8], r2
	umulxhi(%g5, v0, %o5)
	ldx	[rp+16], r3
	mulx	u3, v0, %g4
	umulxhi(u3, v0, %g5)
	addcc	%o3, %o4, %o4
	addxccc(%o5, %g4, %g4)
	addxc(	%g0, %g5, %g5)
	addcc	r1, %o2, r1
	stx	r1, [rp+0]
	addxccc(r2, %o4, r2)
	stx	r2, [rp+8]
	addxccc(r3, %g4, r3)
	stx	r3, [rp+16]
	add	n, -3, n
	add	up, 24, up
	brz	n, L(xit)
	 add	rp, 24, rp
	b	L(com)
	 nop

L(b10):	mulx	%g1, v0, %o4
	ldx	[rp+0], r2
	umulxhi(%g1, v0, %o5)
	ldx	[rp+8], r3
	mulx	%g5, v0, %g4
	umulxhi(%g5, v0, %g5)
	addcc	%o5, %g4, %g4
	addxc(	%g0, %g5, %g5)
	addcc	r2, %o4, r2
	stx	r2, [rp+0]
	addxccc(r3, %g4, r3)
	stx	r3, [rp+8]
	add	n, -2, n
	add	up, 16, up
	brz	n, L(xit)
	 add	rp, 16, rp
	b	L(com)
	 nop

L(b01):	ldx	[rp+0], r3
	mulx	%g1, v0, %g4
	umulxhi(%g1, v0, %g5)
	addcc	r3, %g4, r3
	stx	r3, [rp+0]
	add	n, -1, n
	add	up, 8, up
	brz	n, L(xit)
	 add	rp, 8, rp

L(com):	ldx	[up+0], %g1
L(b0):	ldx	[up+8], u1
	ldx	[up+16], u2
	ldx	[up+24], u3
	mulx	%g1, v0, %o0
	umulxhi(%g1, v0, %o1)
	b	L(lo0)
	 nop

	ALIGN(16)
L(top):	ldx	[up+0], u0
	addxc(	%g0, %g5, %g5)		C propagate carry into carry limb
	ldx	[up+8], u1
	addcc	r0, %o0, r0
	ldx	[up+16], u2
	addxccc(r1, %o2, r1)
	ldx	[up+24], u3
	addxccc(r2, %o4, r2)
	stx	r0, [rp-32]
	addxccc(r3, %g4, r3)
	stx	r1, [rp-24]
	mulx	u0, v0, %o0
	stx	r2, [rp-16]
	umulxhi(u0, v0, %o1)
	stx	r3, [rp-8]
L(lo0):	mulx	u1, v0, %o2
	ldx	[rp+0], r0
	umulxhi(u1, v0, %o3)
	ldx	[rp+8], r1
	mulx	u2, v0, %o4
	ldx	[rp+16], r2
	umulxhi(u2, v0, %o5)
	ldx	[rp+24], r3
	mulx	u3, v0, %g4
	addxccc(%g5, %o0, %o0)
	umulxhi(u3, v0, %g5)
	add	up, 32, up
	addxccc(%o1, %o2, %o2)
	add	rp, 32, rp
	addxccc(%o3, %o4, %o4)
	add	n, -4, n
	addxccc(%o5, %g4, %g4)
	brgz	n, L(top)
	 nop

	addxc(	%g0, %g5, %g5)
	addcc	r0, %o0, r0
	stx	r0, [rp-32]
	addxccc(r1, %o2, r1)
	stx	r1, [rp-24]
	addxccc(r2, %o4, r2)
	stx	r2, [rp-16]
	addxccc(r3, %g4, r3)
	stx	r3, [rp-8]
L(xit):	addxc(	%g0, %g5, %i0)
	ret
	 restore
EPILOGUE()
