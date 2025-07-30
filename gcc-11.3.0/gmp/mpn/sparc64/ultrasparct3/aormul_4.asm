dnl  SPARC v9 mpn_mul_4 and mpn_addmul_4 for T3/T4/T5.

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
C		       mul_4           addmul_4
C UltraSPARC T3:	21.5		22.0
C UltraSPARC T4:	 2.625		 2.75


C The code is well-scheduled and relies on OoO very little.  There is hope that
C this will run at around 2.5 and 2.75 c/l respectively, on T4.

define(`rp', `%i0')
define(`up', `%i1')
define(`n',  `%i2')
define(`vp', `%i3')

define(`v0', `%g1')
define(`v1', `%o7')
define(`v2', `%g2')
define(`v3', `%i3')

define(`w0', `%o0')
define(`w1', `%o1')
define(`w2', `%o2')
define(`w3', `%o3')
define(`w4', `%o4')

define(`r0', `%o5')

define(`u0', `%i4')
define(`u1', `%i5')

define(`rp0', `rp')
define(`rp1', `%g3')
define(`rp2', `%g4')
define(`up0', `up')
define(`up1', `%g5')

ifdef(`OPERATION_mul_4',`
      define(`AM4',      `')
      define(`ADDX',	 `addcc`'$1')
      define(`func',     `mpn_mul_4')
')
ifdef(`OPERATION_addmul_4',`
      define(`AM4',      `$1')
      define(`ADDX',	 `addxccc($1,$2,$3)')
      define(`func',     `mpn_addmul_4')
')


MULFUNC_PROLOGUE(mpn_mul_4 mpn_addmul_4)

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(func)
	save	%sp, -176, %sp

	ldx	[up + 0], u1		C load up[0] early
	andcc	n, 1, %g0		C is n odd?
	ldx	[vp + 0], v0
	sllx	n, 3, n
	ldx	[vp + 8], v1
	add	n, -28, n
	ldx	[vp + 16], v2
	add	rp, -16, rp
	ldx	[vp + 24], v3
	add	up, n, up0
	add	rp, n, rp0
	add	up0, 8, up1
	add	rp0, 8, rp1
	add	rp0, 16, rp2
	mulx	u1, v0, %l0
	mov	0, w0
	mulx	u1, v1, %l1
	mov	0, w1
	mulx	u1, v2, %l2
	mov	0, w2
	mulx	u1, v3, %l3
	mov	0, w3

	be	L(evn)
	 neg	n, n

L(odd):	mov	u1, u0
	ldx	[up1 + n], u1
AM4(`	ldx	[rp2 + n], r0')
	umulxhi(u0, v0, %l4)
	umulxhi(u0, v1, %l5)
	umulxhi(u0, v2, %l6)
	umulxhi(u0, v3, %l7)
	b	L(mid)
	 add	n, 8, n

L(evn):	ldx	[up1 + n], u0
AM4(`	ldx	[rp2 + n], r0')
	umulxhi(u1, v0, %l4)
	umulxhi(u1, v1, %l5)
	umulxhi(u1, v2, %l6)
	umulxhi(u1, v3, %l7)
	add	n, 16, n

	ALIGN(16)
L(top):	addcc	%l0, w0, w0
	mulx	u0, v0, %l0	C w 0
	addxccc(%l1, w1, w1)
	mulx	u0, v1, %l1	C w 1
	addxccc(%l2, w2, w2)
	mulx	u0, v2, %l2	C w 2
	addxccc(%l3, w3, w3)
	mulx	u0, v3, %l3	C w 3
	ldx	[up0 + n], u1
	addxc(	%g0, %g0, w4)
AM4(`	addcc	r0, w0, w0')
	stx	w0, [rp0 + n]
	ADDX(`	%l4, w1, w0')
	umulxhi(u0, v0, %l4)	C w 1
AM4(`	ldx	[rp1 + n], r0')
	addxccc(%l5, w2, w1)
	umulxhi(u0, v1, %l5)	C w 2
	addxccc(%l6, w3, w2)
	umulxhi(u0, v2, %l6)	C w 3
	addxc(	%l7, w4, w3)
	umulxhi(u0, v3, %l7)	C w 4
L(mid):	addcc	%l0, w0, w0
	mulx	u1, v0, %l0	C w 1
	addxccc(%l1, w1, w1)
	mulx	u1, v1, %l1	C w 2
	addxccc(%l2, w2, w2)
	mulx	u1, v2, %l2	C w 3
	addxccc(%l3, w3, w3)
	mulx	u1, v3, %l3	C w 4
	ldx	[up1 + n], u0
	addxc(	%g0, %g0, w4)
AM4(`	addcc	r0, w0, w0')
	stx	w0, [rp1 + n]
	ADDX(`	%l4, w1, w0')
	umulxhi(u1, v0, %l4)	C w 2
AM4(`	ldx	[rp2 + n], r0')
	addxccc(%l5, w2, w1)
	umulxhi(u1, v1, %l5)	C w 3
	addxccc(%l6, w3, w2)
	umulxhi(u1, v2, %l6)	C w 4
	addxc(	%l7, w4, w3)
	umulxhi(u1, v3, %l7)	C w 5
	brlz	n, L(top)
	 add	n, 16, n

L(end):	addcc	%l0, w0, w0
	mulx	u0, v0, %l0
	addxccc(%l1, w1, w1)
	mulx	u0, v1, %l1
	addxccc(%l2, w2, w2)
	mulx	u0, v2, %l2
	addxccc(%l3, w3, w3)
	mulx	u0, v3, %l3
	addxc(	%g0, %g0, w4)
AM4(`	addcc	r0, w0, w0')
	stx	w0, [rp0 + n]
	ADDX(`	%l4, w1, w0')
	umulxhi(u0, v0, %l4)
AM4(`	ldx	[rp1 + n], r0')
	addxccc(%l5, w2, w1)
	umulxhi(u0, v1, %l5)
	addxccc(%l6, w3, w2)
	umulxhi(u0, v2, %l6)
	addxc(	%l7, w4, w3)
	umulxhi(u0, v3, %l7)
	addcc	%l0, w0, w0
	addxccc(%l1, w1, w1)
	addxccc(%l2, w2, w2)
	addxccc(%l3, w3, w3)
	addxc(	%g0, %g0, w4)
AM4(`	addcc	r0, w0, w0')
	stx	w0, [rp1 + n]
	ADDX(`	%l4, w1, w0')
	addxccc(%l5, w2, w1)
	addxccc(%l6, w3, w2)
	stx	w0, [rp2 + n]
	add	n, 16, n
	stx	w1, [rp1 + n]
	stx	w2, [rp2 + n]
	addxc(	%l7, w4, %i0)
	ret
	 restore
EPILOGUE()
