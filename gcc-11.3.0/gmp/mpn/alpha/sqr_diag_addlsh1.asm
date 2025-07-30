dnl  Alpha mpn_sqr_diag_addlsh1.

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

C      cycles/limb
C EV4:      ?
C EV5:     10.2
C EV6:      4.5

C Ideally, one-way code could run at 9 c/l (limited by mulq+umulh) on ev5 and
C about 3.75 c/l on ev6.  Two-way code could run at about 3.25 c/l on ev6.

C Algorithm: We allow ourselves to propagate carry to a product high word
C without worrying for carry out, since (B-1)^2 = B^2-2B+1 has a high word of
C B-2, i.e, will not spill.  We propagate carry similarly to a product low word
C since the problem value B-1 is a quadratic non-residue mod B, but our
C products are squares.

define(`rp',	`r16')
define(`tp',	`r17')
define(`up',	`r18')
define(`n',	`r19')

ASM_START()
PROLOGUE(mpn_sqr_diag_addlsh1)
	ldq	r0, 0(up)
	bis	r31, r31, r21
	bis	r31, r31, r3
	mulq	r0, r0, r7
	stq	r7, 0(rp)
	umulh	r0, r0, r6
	lda	n, -1(n)

	ALIGN(16)
L(top):	ldq	r0, 8(up)
	lda	up, 8(up)
	ldq	r8, 0(tp)
	ldq	r20, 8(tp)
	mulq	r0, r0, r7
	lda	tp, 16(tp)
	sll	r8, 1, r23
	srl	r8, 63, r22
	or	r21, r23, r23
	sll	r20, 1, r24
	addq	r3, r6, r6		C cannot carry per comment above
	or	r22, r24, r24
	addq	r23, r6, r21
	umulh	r0, r0, r6
	cmpult	r21, r23, r1
	addq	r1, r7, r7		C cannot carry per comment above
	stq	r21, 8(rp)
	addq	r24, r7, r22
	stq	r22, 16(rp)
	lda	n, -1(n)
	cmpult	r22, r7, r3
	srl	r20, 63, r21
	lda	rp, 16(rp)
	bne	n, L(top)

	addq	r3, r6, r6		C cannot carry per comment above
	addq	r21, r6, r21
	stq	r21, 8(rp)
	ret	r31, (r26), 1
EPILOGUE()
ASM_END()
