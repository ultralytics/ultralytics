dnl  SPARC v9 mpn_mul_2 and mpn_addmul_2 for T3/T4/T5.

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


C		    cycles/limb      cycles/limb
C		       mul_2           addmul_2
C UltraSPARC T3:	22.5		 23.5
C UltraSPARC T4:	 3.25		 3.75


C The code is reasonably scheduled but also relies on OoO.  There was hope that
C this could run at around 3.0 and 3.5 c/l respectively, on T4.  Two cycles per
C iteration needs to be removed.
C
C We could almost use 2-way unrolling, but currently the wN registers live too
C long.  By changing add x,w1,w1 to add x,w1,w0, i.e. migrate the values down-
C wards, 2-way unrolling should become possible.  With n-indexed addressing it
C should run no slower.
C
C The rp loads to g1/g3 are very much over-scheduled.  Presumably, they could
C be postponed a full way, and then just one register could be used.

C INPUT PARAMETERS
define(`rp', `%i0')
define(`up', `%i1')
define(`n',  `%i2')
define(`vp', `%i3')

define(`v0', `%o0')
define(`v1', `%o1')

define(`w0', `%o2')
define(`w1', `%o3')
define(`w2', `%o4')
define(`w3', `%o5')

ifdef(`OPERATION_mul_2',`
      define(`AM2',      `')
      define(`ADDX',	 `addcc`'$1')
      define(`func',     `mpn_mul_2')
')
ifdef(`OPERATION_addmul_2',`
      define(`AM2',      `$1')
      define(`ADDX',	 `addxccc($1,$2,$3)')
      define(`func',     `mpn_addmul_2')
')


MULFUNC_PROLOGUE(mpn_mul_2 mpn_addmul_2)

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(func)
	save	%sp, -176, %sp

	ldx	[vp+0], v0		C load v0
	and	n, 3, %g5
	ldx	[vp+8], v1		C load v1
	add	n, -6, n
	ldx	[up+0], %g4
	brz	%g5, L(b0)
	 cmp	%g5, 2
	bcs	L(b1)
	 nop
	be	L(b2)
	 nop

L(b3):
AM2(`	ldx	[rp+0], %g1')
	mulx	%g4, v0, w2
	umulxhi(%g4, v0, w3)
	ldx	[up+8], %i5
	mulx	%g4, v1, %l3
	umulxhi(%g4, v1, %l7)
AM2(`	ldx	[rp+8], %g3')
	add	up, -8, up
	add	rp, -8, rp
	b	L(lo3)
	 mov	0, w0

L(b2):
AM2(`	ldx	[rp+0], %g3')
	mulx	%g4, v0, w3
	umulxhi(%g4, v0, w0)
	ldx	[up+8], %i4
	mulx	%g4, v1, %l1
	umulxhi(%g4, v1, %l5)
AM2(`	ldx	[rp+8], %g1')
	add	rp, 16, rp
	brlz	n, L(end)
	 mov	0, w1
	ba	L(top)
	 add	up, 16, up

L(b1):
AM2(`	ldx	[rp+0], %g1')
	mulx	%g4, v0, w0
	umulxhi(%g4, v0, w1)
	ldx	[up+8], %i5
	mulx	%g4, v1, %l3
	umulxhi(%g4, v1, %l7)
AM2(`	ldx	[rp+8], %g3')
	add	up, 8, up
	add	rp, 8, rp
	b	L(lo1)
	 mov	0, w2

L(b0):
AM2(`	ldx	[rp+0], %g3')
	mulx	%g4, v0, w1
	umulxhi(%g4, v0, w2)
	ldx	[up+8], %i4
	mulx	%g4, v1, %l1
	umulxhi(%g4, v1, %l5)
AM2(`	ldx	[rp+8], %g1')
	b	L(lo0)
	 mov	0, w3

	ALIGN(16)			C cycle
L(top):	mulx	%i4, v0, %l2		C 0->5
	umulxhi(%i4, v0, %l6)		C 0->5
	ldx	[up+0], %i5		C 1->6
AM2(`	addcc	w3, %g3, w3')		C 1
	stx	w3, [rp-16]		C 2
	ADDX(`	%l1, w0, w0')		C 2
	addxccc(%l5, w1, w1)		C 3
	mulx	%i4, v1, %l3		C 3->9
	umulxhi(%i4, v1, %l7)		C 4->9
AM2(`	ldx	[rp+0], %g3')		C 4
	addcc	%l2, w0, w0		C 5
	addxccc(%l6, w1, w1)		C 5
	addxc(	%g0, %g0, w2)		C 6
L(lo1):	mulx	%i5, v0, %l0		C 6
	umulxhi(%i5, v0, %l4)		C 7
	ldx	[up+8], %i4		C 7
AM2(`	addcc	w0, %g1, w0')		C 8
	stx	w0, [rp-8]		C 8
	ADDX(`	%l3, w1, w1')		C 9
	addxccc(%l7, w2, w2)		C 9
	mulx	%i5, v1, %l1		C 10
	umulxhi(%i5, v1, %l5)		C 10
AM2(`	ldx	[rp+8], %g1')		C 11
	addcc	%l0, w1, w1		C 11
	addxccc(%l4, w2, w2)		C 12
	addxc(	%g0, %g0, w3)		C 12
L(lo0):	mulx	%i4, v0, %l2		C 13
	umulxhi(%i4, v0, %l6)		C 13
	ldx	[up+16], %i5		C 14
AM2(`	addcc	w1, %g3, w1')		C 14
	stx	w1, [rp+0]		C 15
	ADDX(`	%l1, w2, w2')		C 15
	addxccc(%l5, w3, w3)		C 16
	mulx	%i4, v1, %l3		C 16
	umulxhi(%i4, v1, %l7)		C 17
AM2(`	ldx	[rp+16], %g3')		C 17
	addcc	%l2, w2, w2		C 18
	addxccc(%l6, w3, w3)		C 18
	addxc(	%g0, %g0, w0)		C 19
L(lo3):	mulx	%i5, v0, %l0		C 19
	umulxhi(%i5, v0, %l4)		C 20
	ldx	[up+24], %i4		C 20
AM2(`	addcc	w2, %g1, w2')		C 21
	stx	w2, [rp+8]		C 21
	ADDX(`	%l3, w3, w3')		C 22
	addxccc(%l7, w0, w0)		C 22
	mulx	%i5, v1, %l1		C 23
	umulxhi(%i5, v1, %l5)		C 23
AM2(`	ldx	[rp+24], %g1')		C 24
	addcc	%l0, w3, w3		C 24
	addxccc(%l4, w0, w0)		C 25
	addxc(	%g0, %g0, w1)		C 25
	add	up, 32, up
	add	rp, 32, rp
	brgz	n, L(top)
	 add	n, -4, n

L(end):	mulx	%i4, v0, %l2
	umulxhi(%i4, v0, %l6)
AM2(`	addcc	w3, %g3, w3')
	stx	w3, [rp-16]
	ADDX(`	%l1, w0, w0')
	addxccc(%l5, w1, w1)
	mulx	%i4, v1, %l3
	umulxhi(%i4, v1, %l7)
	addcc	%l2, w0, w0
	addxccc(%l6, w1, w1)
	addxc(	%g0, %g0, w2)
AM2(`	addcc	w0, %g1, w0')
	stx	w0, [rp-8]
	ADDX(`	%l3, w1, w1')
	stx	w1, [rp+0]
	addxc(%l7, w2, %i0)

	ret
	 restore
EPILOGUE()
