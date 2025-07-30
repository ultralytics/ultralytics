dnl  SPARC v9 mpn_submul_1 for T3/T4/T5.

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
C UltraSPARC T4:	 4.5

C INPUT PARAMETERS
define(`rp', `%i0')
define(`up', `%i1')
define(`n',  `%i2')
define(`v0', `%i3')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_submul_1)
	save	%sp, -176, %sp
	ldx	[up+0], %g1

	and	n, 3, %g5
	add	n, -4, n
	brz	%g5, L(b00)
	 cmp	%g5, 2
	bcs	%xcc, L(b01)
	 nop
	bne	%xcc, L(b11)
	 ldx	[up+8], %g4

L(b10):	add	up, 16, up
	addcc	%g0, 0, %g3
	mulx	%g1, v0, %l4
	umulxhi(%g1, v0, %l5)
	ldx	[rp+0], %o2
	mulx	%g4, v0, %l6
	umulxhi(%g4, v0, %l7)
	brlz	n, L(wd2)
	 nop
L(gt2):	ldx	[up+0], %o0
	b	L(lo2)
	 nop

L(b00):	add	rp, -16, rp
	addcc	%g0, 0, %g3
	ldx	[up+8], %o1
	mulx	%g1, v0, %l0
	umulxhi(%g1, v0, %l1)
	ldx	[up+16], %o0
	ldx	[rp+16], %o2
	mulx	%o1, v0, %l2
	umulxhi(%o1, v0, %l3)
	b	     L(lo0)
	 nop

L(b01):	add	up, 8, up
	add	rp, -8, rp
	addcc	%g0, 0, %g3
	ldx	[rp+8], %o3
	mulx	%g1, v0, %l6
	umulxhi(%g1, v0, %l7)
	brlz	n, L(wd1)
	 nop
	ldx	[up+0], %o0
	ldx	[up+8], %o1
	mulx	%o0, v0, %l0
	umulxhi(%o0, v0, %l1)
	b	L(lo1)
	 nop

L(b11):	add	up, 24, up
	add	rp, 8, rp
	addcc	%g0, 0, %g3
	mulx	%g1, v0, %l2
	umulxhi(%g1, v0, %l3)
	ldx	[up-8], %o1
	ldx	[rp-8], %o3
	mulx	%g4, v0, %l4
	umulxhi(%g4, v0, %l5)
	brlz	n, L(end)
	 nop

	ALIGN(16)
L(top):	ldx	[up+0], %o0
	addxccc(%g3, %l2, %g1)
	ldx	[rp+0], %o2
	addxc(	%g0, %l3, %g3)
	mulx	%o1, v0, %l6
	subcc	%o3, %g1, %g4
	umulxhi(%o1, v0, %l7)
	stx	%g4, [rp-8]
L(lo2):	ldx	[up+8], %o1
	addxccc(%g3, %l4, %g1)
	ldx	[rp+8], %o3
	addxc(	%g0, %l5, %g3)
	mulx	%o0, v0, %l0
	subcc	%o2, %g1, %g4
	umulxhi(%o0, v0, %l1)
	stx	%g4, [rp+0]
L(lo1):	ldx	[up+16], %o0
	addxccc(%g3, %l6, %g1)
	ldx	[rp+16], %o2
	addxc(	%g0, %l7, %g3)
	mulx	%o1, v0, %l2
	subcc	%o3, %g1, %g4
	umulxhi(%o1, v0, %l3)
	stx	%g4, [rp+8]
L(lo0):	ldx	[up+24], %o1
	addxccc(%g3, %l0, %g1)
	ldx	[rp+24], %o3
	addxc(	%g0, %l1, %g3)
	mulx	%o0, v0, %l4
	subcc	%o2, %g1, %g4
	umulxhi(%o0, v0, %l5)
	stx	%g4, [rp+16]
	add	n, -4, n
	add	up, 32, up
	brgez	n, L(top)
	 add	rp, 32, rp

L(end):	addxccc(%g3, %l2, %g1)
	ldx	[rp+0], %o2
	addxc(	%g0, %l3, %g3)
	mulx	%o1, v0, %l6
	subcc	%o3, %g1, %g4
	umulxhi(%o1, v0, %l7)
	stx	%g4, [rp-8]
L(wd2):	addxccc(%g3, %l4, %g1)
	ldx	[rp+8], %o3
	addxc(	%g0, %l5, %g3)
	subcc	%o2, %g1, %g4
	stx	%g4, [rp+0]
L(wd1):	addxccc(%g3, %l6, %g1)
	addxc(	%g0, %l7, %g3)
	subcc	%o3, %g1, %g4
	stx	%g4, [rp+8]
	addxc(	%g0, %g3, %i0)
	ret
	 restore
EPILOGUE()
