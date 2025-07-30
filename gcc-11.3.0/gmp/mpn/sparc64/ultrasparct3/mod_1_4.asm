dnl  SPARC T3/T4/T5 mpn_mod_1s_4p.

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
C UltraSPARC T3:	30
C UltraSPARC T4/T5:	 4

C INPUT PARAMETERS
define(`ap',  `%o0')
define(`n',   `%o1')
define(`d',   `%o2')
define(`cps', `%o3')


ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_mod_1s_4p)
	save	%sp, -176, %sp
	ldx	[%i3+16], %o4
	ldx	[%i3+24], %o3
	ldx	[%i3+32], %o2
	ldx	[%i3+40], %o1
	ldx	[%i3+48], %o0

	and	%i1, 3, %g3
	sllx	%i1, 3, %g1
	add	%i0, %g1, %i0
	brz	%g3, L(b00)
	 cmp	%g3, 2
	bcs	%xcc, L(b01)
	 nop
	be	%xcc, L(b10)
	 nop

L(b11):	ldx	[%i0-16], %g2
	mulx	%g2, %o4, %g5
	umulxhi(%g2, %o4, %g3)
	ldx	[%i0-24], %g4
	addcc	%g5, %g4, %g5
	addxc(	%g3, %g0, %g4)
	ldx	[%i0-8], %g2
	mulx	%g2, %o3, %g1
	umulxhi(%g2, %o3, %g3)
	addcc	%g1, %g5, %g1
	addxc(	%g3, %g4, %g2)
	ba,pt	%xcc, .L8
	 add	%i0, -32, %i0

L(b00):	ldx	[%i0-24], %g3
	mulx	%g3, %o4, %g2
	umulxhi(%g3, %o4, %g5)
	ldx	[%i0-32], %g4
	addcc	%g2, %g4, %g2
	addxc(	%g5, %g0, %g3)
	ldx	[%i0-16], %g4
	mulx	%g4, %o3, %g5
	umulxhi(%g4, %o3, %i5)
	addcc	%g2, %g5, %g5
	addxc(	%g3, %i5, %g4)
	ldx	[%i0-8], %g2
	mulx	%g2, %o2, %g1
	umulxhi(%g2, %o2, %g3)
	addcc	%g1, %g5, %g1
	addxc(	%g3, %g4, %g2)
	ba,pt	%xcc, .L8
	 add	%i0, -40, %i0

L(b01):	ldx	[%i0-8], %g1
	mov	0, %g2
	ba,pt	%xcc, .L8
	 add	%i0, -16, %i0

L(b10):	ldx	[%i0-8], %g2
	ldx	[%i0-16], %g1
	add	%i0, -24, %i0

.L8:	add	%i1, -5, %g3
	brlz,pn	%g3, L(end)
	 nop

L(top):	ldx	[%i0-16], %i4
	mulx	%i4, %o4, %o5
	umulxhi(%i4, %o4, %i1)
	ldx	[%i0-24], %i5
	addcc	%o5, %i5, %o5
	addxc(	%i1, %g0, %i4)
	ldx	[%i0-8], %i5
	mulx	%i5, %o3, %o7
	umulxhi(%i5, %o3, %i1)
	addcc	%o5, %o7, %o7
	addxc(	%i4, %i1, %i5)
	ldx	[%i0+0], %g4
	mulx	%g4, %o2, %i1
	umulxhi(%g4, %o2, %i4)
	addcc	%o7, %i1, %i1
	addxc(	%i5, %i4, %g4)
	mulx	%g1, %o1, %i5
	umulxhi(%g1, %o1, %i4)
	addcc	%i1, %i5, %i5
	addxc(	%g4, %i4, %g5)
	mulx	%g2, %o0, %g1
	umulxhi(%g2, %o0, %g4)
	addcc	%g1, %i5, %g1
	addxc(	%g4, %g5, %g2)
	add	%g3, -4, %g3
	brgez,pt %g3, L(top)
	 add	%i0, -32, %i0

L(end):	mulx	%g2, %o4, %g5
	umulxhi(%g2, %o4, %g3)
	addcc	%g1, %g5, %g5
	addxc(	%g3, %g0, %g2)
	ldx	[%i3+8], %i0
	ldx	[%i3], %g4
	sub	%g0, %i0, %i5
	srlx	%g5, %i5, %i5
	sllx	%g2, %i0, %g2
	or	%i5, %g2, %g1
	mulx	%g1, %g4, %l7
	umulxhi(%g1, %g4, %g3)
	sllx	%g5, %i0, %g2
	add	%g1, 1, %g1
	addcc	%l7, %g2, %g5
	addxc(	%g3, %g1, %g1)
	mulx	%g1, %i2, %g1
	sub	%g2, %g1, %g2
	cmp	%g2, %g5
	add	%i2, %g2, %g1
	movlu	%xcc, %g2, %g1
	subcc	%g1, %i2, %g2
	movgeu	%xcc, %g2, %g1
	return	%i7+8
	 srlx	%g1, %o0, %o0
EPILOGUE()

PROLOGUE(mpn_mod_1s_4p_cps)
	save	%sp, -176, %sp
	lzcnt(	%i1, %i5)
	sllx	%i1, %i5, %i1
	call	mpn_invert_limb, 0
	 mov	%i1, %o0
	stx	%o0, [%i0]
	sra	%i5, 0, %g1
	stx	%g1, [%i0+8]
	sub	%g0, %i5, %g2
	srlx	%o0, %g2, %g2
	mov	1, %g1
	sllx	%g1, %i5, %g1
	or	%g2, %g1, %g2
	sub	%g0, %i1, %g1
	mulx	%g2, %g1, %g2
	srlx	%g2, %i5, %g1
	stx	%g1, [%i0+16]

	umulxhi(%o0, %g2, %g3)
	add	%g2, %g3, %g3
	xnor	%g0, %g3, %g3
	mulx	%g3, %i1, %g3
	mulx	%g2, %o0, %g2
	cmp	%g2, %g3
	add	%i1, %g3, %g1
	movgeu	%xcc, %g3, %g1
	srlx	%g1, %i5, %g2
	stx	%g2, [%i0+24]

	umulxhi(%o0, %g1, %g3)
	add	%g1, %g3, %g3
	xnor	%g0, %g3, %g3
	mulx	%g3, %i1, %g3
	mulx	%g1, %o0, %g1
	cmp	%g1, %g3
	add	%i1, %g3, %g2
	movgeu	%xcc, %g3, %g2
	srlx	%g2, %i5, %g1
	stx	%g1, [%i0+32]

	umulxhi(%o0, %g2, %g3)
	add	%g2, %g3, %g3
	xnor	%g0, %g3, %g3
	mulx	%g3, %i1, %g3
	mulx	%g2, %o0, %g2
	cmp	%g2, %g3
	add	%i1, %g3, %g1
	movgeu	%xcc, %g3, %g1
	srlx	%g1, %i5, %g2
	stx	%g2, [%i0+40]

	umulxhi(%o0, %g1, %g2)
	add	%g1, %g2, %g2
	xnor	%g0, %g2, %g2
	mulx	%g2, %i1, %g2
	mulx	%g1, %o0, %o0
	cmp	%o0, %g2
	add	%i1, %g2, %g3
	movgeu	%xcc, %g2, %g3
	srlx	%g3, %i5, %i5
	stx	%i5, [%i0+48]

	return	%i7+8
	 nop
EPILOGUE()
