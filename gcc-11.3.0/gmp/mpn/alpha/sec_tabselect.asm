dnl  Alpha mpn_sec_tabselect.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2011-2013 Free Software Foundation, Inc.

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

C      cycles/limb
C EV4:      ?
C EV5:      2.25
C EV6:      1.64

define(`rp',     `r16')
define(`tp',     `r17')
define(`n',      `r18')
define(`nents',  `r19')
define(`which',  `r20')

define(`i',      `r21')
define(`j',      `r22')
define(`stride', `r23')
define(`mask',   `r24')
define(`k',      `r25')


ASM_START()
PROLOGUE(mpn_sec_tabselect)
	subq	n, 4, j			C outer loop induction variable

	blt	j, L(outer_end)
L(outer_top):
	mov	tp, r8
	lda	r0, 0(r31)
	lda	r1, 0(r31)
	lda	r2, 0(r31)
	lda	r3, 0(r31)
	subq	j, 4, j			C outer loop induction variable
	subq	nents, which, k
	mov	nents, i

	ALIGN(16)
L(top):	ldq	r4, 0(tp)
	ldq	r5, 8(tp)
	cmpeq	k, i, mask
	subq	i, 1, i
	subq	r31, mask, mask
	ldq	r6, 16(tp)
	ldq	r7, 24(tp)
	and	r4, mask, r4
	and	r5, mask, r5
	or	r0, r4, r0
	or	r1, r5, r1
	and	r6, mask, r6
	and	r7, mask, r7
	or	r2, r6, r2
	or	r3, r7, r3
	s8addq	n, tp, tp
	bne	i, L(top)

	stq	r0, 0(rp)
	stq	r1, 8(rp)
	stq	r2, 16(rp)
	stq	r3, 24(rp)
	addq	r8, 32, tp
	addq	rp, 32, rp
	bge	j, L(outer_top)
L(outer_end):

	and	n, 2, r0
	beq	r0, L(b0x)
L(b1x):	mov	tp, r8
	lda	r0, 0(r31)
	lda	r1, 0(r31)
	subq	nents, which, k
	mov	nents, i
	ALIGN(16)
L(tp2):	ldq	r4, 0(tp)
	ldq	r5, 8(tp)
	cmpeq	k, i, mask
	subq	i, 1, i
	subq	r31, mask, mask
	and	r4, mask, r4
	and	r5, mask, r5
	or	r0, r4, r0
	or	r1, r5, r1
	s8addq	n, tp, tp
	bne	i, L(tp2)
	stq	r0, 0(rp)
	stq	r1, 8(rp)
	addq	r8, 16, tp
	addq	rp, 16, rp

L(b0x):	and	n, 1, r0
	beq	r0, L(b00)
L(b01):	lda	r0, 0(r31)
	subq	nents, which, k
	mov	nents, i
	ALIGN(16)
L(tp1):	ldq	r4, 0(tp)
	cmpeq	k, i, mask
	subq	i, 1, i
	subq	r31, mask, mask
	and	r4, mask, r4
	or	r0, r4, r0
	s8addq	n, tp, tp
	bne	i, L(tp1)
	stq	r0, 0(rp)

L(b00):	ret	r31, (r26), 1
EPILOGUE()
