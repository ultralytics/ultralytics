dnl  Alpha mpn_bdiv_dbm1c.

dnl  Copyright 2008 Free Software Foundation, Inc.

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
C EV4:     42
C EV5:     18
C EV6:      3

C TODO
C  * Try less unrolling, 2-way should give the same performance.
C  * Optimize feed-in and wind-down code, for speed, and perhaps further for
C    code size.
C  * This runs optimally given the algorithm, r8 is on a 3 operation recurrency
C    path.  We have not tried very hard to find a better algorithm.  Perhaps
C    it would be a good task for the GNU superoptimizer.

C INPUT PARAMETERS
define(`rp', `r16')
define(`up', `r17')
define(`n',  `r18')
define(`bd', `r19')
define(`cy', `r19')


ASM_START()
PROLOGUE(mpn_bdiv_dbm1c)
	mov	r20, r8

	ldq	r24, 0(r17)
	and	r18, 3, r28
	lda	r18, -4(r18)
	beq	r28, L(b0)
	cmpeq	r28, 1, r21
	bne	r21, L(b1)
	cmpeq	r28, 2, r21
	bne	r21, L(b2)


L(b3):	ldq	r2, 8(r17)
	ldq	r3, 16(r17)
	bgt	r18, L(gt3)

	mulq	r24, r19, r5	C U1
	umulh	r24, r19, r21	C U1
	mulq	r2, r19, r6	C U1
	umulh	r2, r19, r22	C U1
	mulq	r3, r19, r7	C U1
	umulh	r3, r19, r23	C U1
	lda	r16, -32(r16)
	br	L(cj3)

L(gt3):	ldq	r0, 24(r17)
	mulq	r24, r19, r5	C U1
	umulh	r24, r19, r21	C U1
	ldq	r1, 32(r17)
	mulq	r2, r19, r6	C U1
	umulh	r2, r19, r22	C U1
	ldq	r2, 40(r17)
	mulq	r3, r19, r7	C U1
	umulh	r3, r19, r23	C U1
	ldq	r3, 48(r17)
	lda	r18, -4(r18)
	lda	r17, 56(r17)
	mulq	r0, r19, r4	C U1
	bgt	r18, L(L3)

	br	L(cj7)


L(b2):	ldq	r3, 8(r17)
	bgt	r18, L(gt2)

	mulq	r24, r19, r6	C U1
	umulh	r24, r19, r22	C U1
	mulq	r3, r19, r7	C U1
	umulh	r3, r19, r23	C U1
	lda	r16, -40(r16)
	br	L(cj2)

L(gt2):	ldq	r0, 16(r17)
	ldq	r1, 24(r17)
	mulq	r24, r19, r6	C U1
	umulh	r24, r19, r22	C U1
	ldq	r2, 32(r17)
	mulq	r3, r19, r7	C U1
	umulh	r3, r19, r23	C U1
	ldq	r3, 40(r17)
	lda	r18, -4(r18)
	lda	r17, 48(r17)
	mulq	r0, r19, r4	C U1
	umulh	r0, r19, r20	C U1
	lda	r16, -8(r16)
	bgt	r18, L(gt6)

	mulq	r1, r19, r5	C U1
	br	L(cj6)

L(gt6):	ldq	r0, 0(r17)
	mulq	r1, r19, r5	C U1
	br	L(L2)


L(b1):	bgt	r18, L(gt1)

	mulq	r24, r19, r7	C U1
	umulh	r24, r19, r23	C U1
	lda	r16, -48(r16)
	br	L(cj1)

L(gt1):	ldq	r0, 8(r17)
	ldq	r1, 16(r17)
	ldq	r2, 24(r17)
	mulq	r24, r19, r7	C U1
	umulh	r24, r19, r23	C U1
	ldq	r3, 32(r17)
	lda	r18, -4(r18)
	lda	r17, 40(r17)
	mulq	r0, r19, r4	C U1
	umulh	r0, r19, r20	C U1
	lda	r16, -16(r16)
	bgt	r18, L(gt5)

	mulq	r1, r19, r5	C U1
	umulh	r1, r19, r21	C U1
	mulq	r2, r19, r6	C U1
	br	L(cj5)

L(gt5):	ldq	r0, 0(r17)
	mulq	r1, r19, r5	C U1
	umulh	r1, r19, r21	C U1
	ldq	r1, 8(r17)
	mulq	r2, r19, r6	C U1
	br	L(L1)


L(b0):	ldq	r1, 8(r17)
	ldq	r2, 16(r17)
	ldq	r3, 24(r17)
	lda	r17, 32(r17)
	lda	r16, -24(r16)
	mulq	r24, r19, r4	C U1
	umulh	r24, r19, r20	C U1
	bgt	r18, L(gt4)

	mulq	r1, r19, r5	C U1
	umulh	r1, r19, r21	C U1
	mulq	r2, r19, r6	C U1
	umulh	r2, r19, r22	C U1
	mulq	r3, r19, r7	C U1
	br	L(cj4)

L(gt4):	ldq	r0, 0(r17)
	mulq	r1, r19, r5	C U1
	umulh	r1, r19, r21	C U1
	ldq	r1, 8(r17)
	mulq	r2, r19, r6	C U1
	umulh	r2, r19, r22	C U1
	ldq	r2, 16(r17)
	mulq	r3, r19, r7	C U1
	br	L(L0)

C *** MAIN LOOP START ***
	ALIGN(16)
L(top):	mulq	r0, r19, r4	C U1
	subq	r8, r28, r8
L(L3):	umulh	r0, r19, r20	C U1
	cmpult	r8, r5, r28
	ldq	r0, 0(r17)
	subq	r8, r5, r8
	addq	r21, r28, r28
	stq	r8, 0(r16)

	mulq	r1, r19, r5	C U1
	subq	r8, r28, r8
L(L2):	umulh	r1, r19, r21	C U1
	cmpult	r8, r6, r28
	ldq	r1, 8(r17)
	subq	r8, r6, r8
	addq	r22, r28, r28
	stq	r8, 8(r16)

	mulq	r2, r19, r6	C U1
	subq	r8, r28, r8
L(L1):	umulh	r2, r19, r22	C U1
	cmpult	r8, r7, r28
	ldq	r2, 16(r17)
	subq	r8, r7, r8
	addq	r23, r28, r28
	stq	r8, 16(r16)

	mulq	r3, r19, r7	C U1
	subq	r8, r28, r8
L(L0):	umulh	r3, r19, r23	C U1
	cmpult	r8, r4, r28
	ldq	r3, 24(r17)
	subq	r8, r4, r8
	addq	r20, r28, r28
	stq	r8, 24(r16)

	lda	r18, -4(r18)
	lda	r17, 32(r17)
	lda	r16, 32(r16)
	bgt	r18, L(top)
C *** MAIN LOOP END ***

	mulq	r0, r19, r4	C U1
	subq	r8, r28, r8
L(cj7):	umulh	r0, r19, r20	C U1
	cmpult	r8, r5, r28
	subq	r8, r5, r8
	addq	r21, r28, r28
	stq	r8, 0(r16)
	mulq	r1, r19, r5	C U1
	subq	r8, r28, r8
L(cj6):	umulh	r1, r19, r21	C U1
	cmpult	r8, r6, r28
	subq	r8, r6, r8
	addq	r22, r28, r28
	stq	r8, 8(r16)
	mulq	r2, r19, r6	C U1
	subq	r8, r28, r8
L(cj5):	umulh	r2, r19, r22	C U1
	cmpult	r8, r7, r28
	subq	r8, r7, r8
	addq	r23, r28, r28
	stq	r8, 16(r16)
	mulq	r3, r19, r7	C U1
	subq	r8, r28, r8
L(cj4):	umulh	r3, r19, r23	C U1
	cmpult	r8, r4, r28
	subq	r8, r4, r8
	addq	r20, r28, r28
	stq	r8, 24(r16)
	subq	r8, r28, r8
L(cj3):	cmpult	r8, r5, r28
	subq	r8, r5, r8
	addq	r21, r28, r28
	stq	r8, 32(r16)
	subq	r8, r28, r8
L(cj2):	cmpult	r8, r6, r28
	subq	r8, r6, r8
	addq	r22, r28, r28
	stq	r8, 40(r16)
	subq	r8, r28, r8
L(cj1):	cmpult	r8, r7, r28
	subq	r8, r7, r8
	addq	r23, r28, r28
	stq	r8, 48(r16)
	subq	r8, r28, r0
	ret	r31, (r26), 1

EPILOGUE()
ASM_END()
