dnl  SPARC v9 mpn_mul_1 for T3/T4/T5.

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
C UltraSPARC T3:	23
C UltraSPARC T4:	 3

C INPUT PARAMETERS
define(`rp', `%i0')
define(`up', `%i1')
define(`n',  `%i2')
define(`v0', `%i3')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_mul_1)
	save	%sp, -176, %sp

	and	n, 3, %g5
	add	n, -4, n
	brz	%g5, L(b0)
	 cmp	%g5, 2
	bcs	%xcc, L(b1)
	 nop
	be	%xcc, L(b2)
	 nop

L(b3):	addcc	%g0, %g0, %i5
	ldx	[up+0], %l0
	ldx	[up+8], %l1
	ldx	[up+16], %l2
	mulx	%l0, v0, %o0
	umulxhi(%l0, v0, %o1)
	brgz	n, L(gt3)
	 add	rp, -8, rp
	mulx	%l1, v0, %o2
	umulxhi(%l1, v0, %o3)
	b	L(wd3)
	 nop
L(gt3):	ldx	[up+24], %l3
	mulx	%l1, v0, %o2
	umulxhi(%l1, v0, %o3)
	add	up, 24, up
	b	L(lo3)
	 add	n, -3, n

L(b2):	addcc	%g0, %g0, %o1
	ldx	[up+0], %l1
	ldx	[up+8], %l2
	brgz	n, L(gt2)
	 add	rp, -16, rp
	mulx	%l1, v0, %o2
	umulxhi(%l1, v0, %o3)
	mulx	%l2, v0, %o4
	umulxhi(%l2, v0, %o5)
	b	L(wd2)
	 nop
L(gt2):	ldx	[up+16], %l3
	mulx	%l1, v0, %o2
	umulxhi(%l1, v0, %o3)
	ldx	[up+24], %l0
	mulx	%l2, v0, %o4
	umulxhi(%l2, v0, %o5)
	add	up, 16, up
	b	L(lo2)
	 add	n, -2, n

L(b1):	addcc	%g0, %g0, %o3
	ldx	[up+0], %l2
	brgz	n, L(gt1)
	nop
	mulx	%l2, v0, %o4
	stx	%o4, [rp+0]
	umulxhi(%l2, v0, %i0)
	ret
	 restore
L(gt1):	ldx	[up+8], %l3
	ldx	[up+16], %l0
	mulx	%l2, v0, %o4
	umulxhi(%l2, v0, %o5)
	ldx	[up+24], %l1
	mulx	%l3, v0, %i4
	umulxhi(%l3, v0, %i5)
	add	rp, -24, rp
	add	up, 8, up
	b	L(lo1)
	 add	n, -1, n

L(b0):	addcc	%g0, %g0, %o5
	ldx	[up+0], %l3
	ldx	[up+8], %l0
	ldx	[up+16], %l1
	mulx	%l3, v0, %i4
	umulxhi(%l3, v0, %i5)
	ldx	[up+24], %l2
	mulx	%l0, v0, %o0
	umulxhi(%l0, v0, %o1)
	b	L(lo0)
	 nop

	ALIGN(16)
L(top):	ldx	[up+0], %l3	C 0
	addxccc(%i4, %o5, %i4)	C 0
	mulx	%l1, v0, %o2	C 1
	stx	%i4, [rp+0]	C 1
	umulxhi(%l1, v0, %o3)	C 2
L(lo3):	ldx	[up+8], %l0	C 2
	addxccc(%o0, %i5, %o0)	C 3
	mulx	%l2, v0, %o4	C 3
	stx	%o0, [rp+8]	C 4
	umulxhi(%l2, v0, %o5)	C 4
L(lo2):	ldx	[up+16], %l1	C 5
	addxccc(%o2, %o1, %o2)	C 5
	mulx	%l3, v0, %i4	C 6
	stx	%o2, [rp+16]	C 6
	umulxhi(%l3, v0, %i5)	C 7
L(lo1):	ldx	[up+24], %l2	C 7
	addxccc(%o4, %o3, %o4)	C 8
	mulx	%l0, v0, %o0	C 8
	stx	%o4, [rp+24]	C 9
	umulxhi(%l0, v0, %o1)	C 9
	add	rp, 32, rp	C 10
L(lo0):	add	up, 32, up	C 10
	brgz	n, L(top)	C 11
	 add	n, -4, n	C 11

L(end):	addxccc(%i4, %o5, %i4)
	mulx	%l1, v0, %o2
	stx	%i4, [rp+0]
	umulxhi(%l1, v0, %o3)
	addxccc(%o0, %i5, %o0)
L(wd3):	mulx	%l2, v0, %o4
	stx	%o0, [rp+8]
	umulxhi(%l2, v0, %o5)
	addxccc(%o2, %o1, %o2)
L(wd2):	stx	%o2, [rp+16]
	addxccc(%o4, %o3, %o4)
	stx	%o4, [rp+24]
	addxc(	%g0, %o5, %i0)
	ret
	 restore
EPILOGUE()
