dnl  SPARC T3/T4/T5 mpn_bdiv_dbm1c.

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

C                  cycles/limb
C UltraSPARC T3:	25
C UltraSPARC T4/T5:	 4

C INPUT PARAMETERS
define(`qp',  `%i0')
define(`ap',  `%i1')
define(`n',   `%i2')
define(`bd',  `%i3')
define(`h',   `%i4')

define(`plo0',`%g4')  define(`plo1',`%g5')
define(`phi0',`%l0')  define(`phi1',`%l1')
define(`a0',  `%g1')  define(`a1',  `%g3')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_bdiv_dbm1c)
	save	%sp, -176, %sp

	and	n, 3, %g5
	ldx	[ap + 0], %g2
	add	n, -5, n
	brz	%g5, L(b0)
	 cmp	%g5, 2
	bcs	%xcc, L(b1)
	 nop
	be	%xcc, L(b2)
	 nop

L(b3):	ldx	[ap + 8], a0
	mulx	bd, %g2, plo1
	umulxhi(bd, %g2, phi1)
	ldx	[ap + 16], a1
	add	qp, -24, qp
	b	L(lo3)
	 add	ap, -8, ap

L(b2):	ldx	[ap + 8], a1
	mulx	bd, %g2, plo0
	umulxhi(bd, %g2, phi0)
	brlz,pt n, L(wd2)
	 nop
L(gt2):	ldx	[ap + 16], a0
	add	ap, 16, ap
	b	L(lo2)
	 add	n, -1, n

L(b1):	mulx	bd, %g2, plo1
	 umulxhi(bd, %g2, phi1)
	brlz,pn	n, L(wd1)
	 add	qp, -8, qp
L(gt1):	ldx	[ap + 8], a0
	ldx	[ap + 16], a1
	b	L(lo1)
	 add	ap, 8, ap

L(b0):	ldx	[ap + 8], a1
	mulx	bd, %g2, plo0
	umulxhi(bd, %g2, phi0)
	ldx	[ap + 16], a0
	b	L(lo0)
	 add	qp, -16, qp

L(top):	ldx	[ap + 0], a0
	sub	h, phi1, h
L(lo2):	mulx	bd, a1, plo1
	umulxhi(bd, a1, phi1)
	subcc	h, plo0, h
	addxc(	phi0, %g0, phi0)
	stx	h, [qp + 0]
	ldx	[ap + 8], a1
	sub	h, phi0, h
L(lo1):	mulx	bd, a0, plo0
	umulxhi(bd, a0, phi0)
	subcc	h, plo1, h
	addxc(	phi1, %g0, phi1)
	stx	h, [qp + 8]
	ldx	[ap + 16], a0
	sub	h, phi1, h
L(lo0):	mulx	bd, a1, plo1
	umulxhi(bd, a1, phi1)
	subcc	h, plo0, h
	addxc(	phi0, %g0, phi0)
	stx	h, [qp + 16]
	ldx	[ap + 24], a1
	sub	h, phi0, h
L(lo3):	mulx	bd, a0, plo0
	umulxhi(bd, a0, phi0)
	subcc	h, plo1, h
	addxc(	phi1, %g0, phi1)
	stx	h, [qp + 24]
	add	ap, 32, ap
	add	qp, 32, qp
	brgz,pt	n, L(top)
	 add	n, -4, n

L(end):	sub	h, phi1, h
L(wd2):	mulx	bd, a1, plo1
	umulxhi(bd, a1, phi1)
	subcc	h, plo0, h
	addxc(	phi0, %g0, phi0)
	stx	h, [qp + 0]
	sub	h, phi0, h
L(wd1):	subcc	h, plo1, h
	addxc(	phi1, %g0, phi1)
	stx	h, [qp + 8]
	sub	h, phi1, %i0

	ret
	 restore
EPILOGUE()
