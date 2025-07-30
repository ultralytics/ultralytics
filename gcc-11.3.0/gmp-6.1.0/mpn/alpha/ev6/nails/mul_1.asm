dnl  Alpha ev6 nails mpn_mul_1.

dnl  Copyright 2002, 2005, 2006 Free Software Foundation, Inc.

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
C EV4:    42
C EV5:    18
C EV6:     3.25

C TODO
C  * Reroll loop for 3.0 c/l with current 4-way unrolling.
C  * The loop is overscheduled wrt loads and wrt multiplies, in particular
C    umulh.
C  * Use FP loop count and multiple exit points, that would simplify feed-in lp0
C    and would work since the loop structure is really regular.

C  INPUT PARAMETERS
define(`rp',`r16')
define(`up',`r17')
define(`n', `r18')
define(`vl0',`r19')

define(`numb_mask',`r6')

define(`m0a',`r0')
define(`m0b',`r1')
define(`m1a',`r2')
define(`m1b',`r3')
define(`m2a',`r20')
define(`m2b',`r21')
define(`m3a',`r22')
define(`m3b',`r23')

define(`acc0',`r25')
define(`acc1',`r27')

define(`ul0',`r4')
define(`ul1',`r5')
define(`ul2',`r4')
define(`ul3',`r5')

define(`rl0',`r24')
define(`rl1',`r24')
define(`rl2',`r24')
define(`rl3',`r24')

define(`t0',`r7')
define(`t1',`r8')

define(`NAIL_BITS',`GMP_NAIL_BITS')
define(`NUMB_BITS',`GMP_NUMB_BITS')

dnl  This declaration is munged by configure
NAILS_SUPPORT(1-63)

ASM_START()
PROLOGUE(mpn_mul_1)
	sll	vl0, NAIL_BITS, vl0
	lda	numb_mask, -1(r31)
	srl	numb_mask, NAIL_BITS, numb_mask

	and	n,	3,	r25
	cmpeq	r25,	1,	r21
	bne	r21,	L(1m4)
	cmpeq	r25,	2,	r21
	bne	r21,	L(2m4)
	beq	r25,	L(0m4)

L(3m4):	ldq	ul3,	0(up)
	lda	n,	-4(n)
	ldq	ul0,	8(up)
	mulq	vl0,	ul3,	m3a
	umulh	vl0,	ul3,	m3b
	ldq	ul1,	16(up)
	lda	up,	24(up)
	lda	rp,	-8(rp)
	mulq	vl0,	ul0,	m0a
	umulh	vl0,	ul0,	m0b
	bge	n,	L(ge3)

	mulq	vl0,	ul1,	m1a
	umulh	vl0,	ul1,	m1b
	srl	m3a,NAIL_BITS,	t0
	addq	t0,	r31,	acc1
	srl	m0a,NAIL_BITS,	t0
	addq	t0,	m3b,	acc0
	srl	acc1,NUMB_BITS,	t1
	br	r31,	L(ta3)

L(ge3):	ldq	ul2,	0(up)
	mulq	vl0,	ul1,	m1a
	umulh	vl0,	ul1,	m1b
	srl	m3a,NAIL_BITS,	t0
	ldq	ul3,	8(up)
	lda	n,	-4(n)
	mulq	vl0,	ul2,	m2a
	addq	t0,	r31,	acc1
	umulh	vl0,	ul2,	m2b
	srl	m0a,NAIL_BITS,	t0
	ldq	ul0,	16(up)
	mulq	vl0,	ul3,	m3a
	addq	t0,	m3b,	acc0
	srl	acc1,NUMB_BITS,	t1
	br	r31,	L(el3)

L(0m4):	lda	n,	-8(n)
	ldq	ul2,	0(up)
	ldq	ul3,	8(up)
	mulq	vl0,	ul2,	m2a
	umulh	vl0,	ul2,	m2b
	ldq	ul0,	16(up)
	mulq	vl0,	ul3,	m3a
	umulh	vl0,	ul3,	m3b
	ldq	ul1,	24(up)
	lda	up,	32(up)
	mulq	vl0,	ul0,	m0a
	umulh	vl0,	ul0,	m0b
	bge	n,	L(ge4)

	srl	m2a,NAIL_BITS,	t0
	mulq	vl0,	ul1,	m1a
	addq	t0,	r31,	acc0
	umulh	vl0,	ul1,	m1b
	srl	m3a,NAIL_BITS,	t0
	addq	t0,	m2b,	acc1
	srl	acc0,NUMB_BITS,	t1
	br	r31,	L(ta4)

L(ge4):	srl	m2a,NAIL_BITS,	t0
	ldq	ul2,	0(up)
	mulq	vl0,	ul1,	m1a
	addq	t0,	r31,	acc0
	umulh	vl0,	ul1,	m1b
	srl	m3a,NAIL_BITS,	t0
	ldq	ul3,	8(up)
	lda	n,	-4(n)
	mulq	vl0,	ul2,	m2a
	addq	t0,	m2b,	acc1
	srl	acc0,NUMB_BITS,	t1
	br	r31,	L(el0)

L(2m4):	lda	n,	-4(n)
	ldq	ul0,	0(up)
	ldq	ul1,	8(up)
	lda	up,	16(up)
	lda	rp,	-16(rp)
	mulq	vl0,	ul0,	m0a
	umulh	vl0,	ul0,	m0b
	bge	n,	L(ge2)

	mulq	vl0,	ul1,	m1a
	umulh	vl0,	ul1,	m1b
	srl	m0a,NAIL_BITS,	t0
	addq	t0,	r31,	acc0
	srl	m1a,NAIL_BITS,	t0
	addq	t0,	m0b,	acc1
	srl	acc0,NUMB_BITS,	t1
	br	r31,	L(ta2)

L(ge2):	ldq	ul2,	0(up)
	mulq	vl0,	ul1,	m1a
	umulh	vl0,	ul1,	m1b
	ldq	ul3,	8(up)
	lda	n,	-4(n)
	mulq	vl0,	ul2,	m2a
	umulh	vl0,	ul2,	m2b
	srl	m0a,NAIL_BITS,	t0
	ldq	ul0,	16(up)
	mulq	vl0,	ul3,	m3a
	addq	t0,	r31,	acc0
	umulh	vl0,	ul3,	m3b
	srl	m1a,NAIL_BITS,	t0
	ldq	ul1,	24(up)
	lda	up,	32(up)
	lda	rp,	32(rp)
	mulq	vl0,	ul0,	m0a
	addq	t0,	m0b,	acc1
	srl	acc0,NUMB_BITS,	t1
	bge	n,	L(el2)

	br	r31,	L(ta6)

L(1m4):	lda	n,	-4(n)
	ldq	ul1,	0(up)
	lda	up,	8(up)
	lda	rp,	-24(rp)
	bge	n,	L(ge1)

	mulq	vl0,	ul1,	m1a
	umulh	vl0,	ul1,	m1b
	srl	m1a,NAIL_BITS,	t0
	addq	t0,	r31,	acc1
	and	acc1,numb_mask,	r28
	srl	acc1,NUMB_BITS,	t1
	stq	r28,	24(rp)
	addq	t1,	m1b,	r0
	ret	r31,	(r26),	1

L(ge1):	ldq	ul2,	0(up)
	mulq	vl0,	ul1,	m1a
	umulh	vl0,	ul1,	m1b
	ldq	ul3,	8(up)
	lda	n,	-4(n)
	mulq	vl0,	ul2,	m2a
	umulh	vl0,	ul2,	m2b
	ldq	ul0,	16(up)
	mulq	vl0,	ul3,	m3a
	umulh	vl0,	ul3,	m3b
	srl	m1a,NAIL_BITS,	t0
	ldq	ul1,	24(up)
	lda	up,	32(up)
	lda	rp,	32(rp)
	mulq	vl0,	ul0,	m0a
	addq	t0,	r31,	acc1
	umulh	vl0,	ul0,	m0b
	srl	m2a,NAIL_BITS,	t0
	mulq	vl0,	ul1,	m1a
	addq	t0,	m1b,	acc0
	srl	acc1,NUMB_BITS,	t1
	blt	n,	L(ta5)

L(ge5):	ldq	ul2,	0(up)
	br	r31,	L(el1)

	ALIGN(16)
L(top):	mulq	vl0,	ul0,	m0a		C U1
	addq	t0,	m0b,	acc1		C L0
	srl	acc0,NUMB_BITS,	t1		C U0
	stq	r28,	-24(rp)			C L1
C
L(el2):	umulh	vl0,	ul0,	m0b		C U1
	and	acc0,numb_mask,	r28		C L0
	unop					C U0
	unop					C L1
C
	unop					C U1
	addq	t1,	acc1,	acc1		C L0
	srl	m2a,NAIL_BITS,	t0		C U0
	ldq	ul2,	0(up)			C L1
C
	mulq	vl0,	ul1,	m1a		C U1
	addq	t0,	m1b,	acc0		C L0
	srl	acc1,NUMB_BITS,	t1		C U0
	stq	r28,	-16(rp)			C L1
C
L(el1):	umulh	vl0,	ul1,	m1b		C U1
	and	acc1,numb_mask,	r28		C L0
	unop					C U0
	lda	n,	-4(n)			C L1
C
	unop					C U1
	addq	t1,	acc0,	acc0		C L0
	srl	m3a,NAIL_BITS,	t0		C U0
	ldq	ul3,	8(up)			C L1
C
	mulq	vl0,	ul2,	m2a		C U1
	addq	t0,	m2b,	acc1		C L0
	srl	acc0,NUMB_BITS,	t1		C U0
	stq	r28,	-8(rp)			C L1
C
L(el0):	umulh	vl0,	ul2,	m2b		C U1
	and	acc0,numb_mask,	r28		C L0
	unop					C U0
	unop					C L1
C
	unop					C U1
	addq	t1,	acc1,	acc1		C L0
	srl	m0a,NAIL_BITS,	t0		C U0
	ldq	ul0,	16(up)			C L1
C
	mulq	vl0,	ul3,	m3a		C U1
	addq	t0,	m3b,	acc0		C L0
	srl	acc1,NUMB_BITS,	t1		C U0
	stq	r28,	0(rp)			C L1
C
L(el3):	umulh	vl0,	ul3,	m3b		C U1
	and	acc1,numb_mask,	r28		C L0
	unop					C U0
	unop					C L1
C
	unop					C U1
	addq	t1,	acc0,	acc0		C L0
	srl	m1a,NAIL_BITS,	t0		C U0
	ldq	ul1,	24(up)			C L1
C
	lda	up,	32(up)			C L0
	unop					C U1
	lda	rp,	32(rp)			C L1
	bge	n,	L(top)			C U0

L(end):	mulq	vl0,	ul0,	m0a
	addq	t0,	m0b,	acc1
	srl	acc0,NUMB_BITS,	t1
	stq	r28,	-24(rp)
L(ta6):	umulh	vl0,	ul0,	m0b
	and	acc0,numb_mask,	r28
	addq	t1,	acc1,	acc1
	srl	m2a,NAIL_BITS,	t0
	mulq	vl0,	ul1,	m1a
	addq	t0,	m1b,	acc0
	srl	acc1,NUMB_BITS,	t1
	stq	r28,	-16(rp)
L(ta5):	umulh	vl0,	ul1,	m1b
	and	acc1,numb_mask,	r28
	addq	t1,	acc0,	acc0
	srl	m3a,NAIL_BITS,	t0
	addq	t0,	m2b,	acc1
	srl	acc0,NUMB_BITS,	t1
	stq	r28,	-8(rp)
	ALIGN(16)
L(ta4):	and	acc0,numb_mask,	r28
	addq	t1,	acc1,	acc1
	srl	m0a,NAIL_BITS,	t0
	addq	t0,	m3b,	acc0
	srl	acc1,NUMB_BITS,	t1
	stq	r28,	0(rp)
	unop
	ALIGN(16)
L(ta3):	and	acc1,numb_mask,	r28
	addq	t1,	acc0,	acc0
	srl	m1a,NAIL_BITS,	t0
	addq	t0,	m0b,	acc1
	srl	acc0,NUMB_BITS,	t1
	stq	r28,	8(rp)
	unop
	ALIGN(16)
L(ta2):	and	acc0,numb_mask,	r28
	addq	t1,	acc1,	acc1
	srl	acc1,NUMB_BITS,	t1
	stq	r28,	16(rp)
	and	acc1,numb_mask,	r28
	addq	t1,	m1b,	r0
	stq	r28,	24(rp)
	ret	r31,	(r26),	1
EPILOGUE()
ASM_END()
