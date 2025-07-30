dnl  SPARC T3/T4/T5 mpn_divexact_1.

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
C UltraSPARC T3:	31
C UltraSPARC T4/T5:	20-26  hits 20 early, then sharply drops

C INPUT PARAMETERS
define(`qp',  `%i0')
define(`ap',  `%i1')
define(`n',   `%i2')
define(`d',   `%i3')

define(`dinv',`%o4')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_divexact_1)
	save	%sp, -176, %sp
	cmp	n, 1
	bne,pt	%xcc, L(gt1)
	 ldx	[ap], %o5
	udivx	%o5, d, %g1
	stx	%g1, [qp]
	return	%i7+8
	 nop

L(gt1):	add	d, -1, %g1
	andn	%g1, d, %g1
	popc	%g1, %i4		C i4 = count_trailing_zeros(d)

	srlx	d, %i4, d
	srlx	d, 1, %g1
	and	%g1, 127, %g1

	LEA64(binvert_limb_table, g2, g4)
	ldub	[%g2+%g1], %g1
	add	%g1, %g1, %g2
	mulx	%g1, %g1, %g1
	mulx	%g1, d, %g1
	sub	%g2, %g1, %g2
	add	%g2, %g2, %g1
	mulx	%g2, %g2, %g2
	mulx	%g2, d, %g2
	sub	%g1, %g2, %g1
	add	%g1, %g1, %o7
	mulx	%g1, %g1, %g1
	mulx	%g1, d, %g1
	add	n, -2, n
	brz,pt	%i4, L(norm)
	 sub	%o7, %g1, dinv

L(unnorm):
	mov	0, %g4
	sub	%g0, %i4, %o2
	srlx	%o5, %i4, %o5
L(top_unnorm):
	ldx	[ap+8], %g3
	add	ap, 8, ap
	sllx	%g3, %o2, %g5
	or	%g5, %o5, %g5
	srlx	%g3, %i4, %o5
	subcc	%g5, %g4, %g4
	mulx	%g4, dinv, %g1
	stx	%g1, [qp]
	add	qp, 8, qp
	umulxhi(d, %g1, %g1)
	addxc(	%g1, %g0, %g4)
	brgz,pt	n, L(top_unnorm)
	 add	n, -1, n

	sub	%o5, %g4, %g4
	mulx	%g4, dinv, %g1
	stx	%g1, [qp]
	return	%i7+8
	 nop

L(norm):
	mulx	dinv, %o5, %g1
	stx	%g1, [qp]
	add	qp, 8, qp
	addcc	%g0, 0, %g4
L(top_norm):
	umulxhi(d, %g1, %g1)
	ldx	[ap+8], %g5
	add	ap, 8, ap
	addxc(	%g1, %g0, %g1)
	subcc	%g5, %g1, %g1
	mulx	%g1, dinv, %g1
	stx	%g1, [qp]
	add	qp, 8, qp
	brgz,pt	n, L(top_norm)
	 add	n, -1, n

	return	%i7+8
	 nop
EPILOGUE()
