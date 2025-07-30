dnl  Alpha ev6 nails mpn_addmul_4.

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

C Runs at 2.5 cycles/limb.

C We should go for 2-way unrolling over 17 cycles, for 2.125 c/l corresponding
C to 3.24 insn/cycle.


C  INPUT PARAMETERS
define(`rp',`r16')
define(`up',`r17')
define(`n',`r18')
define(`vp',`r19')

C  Useful register aliases
define(`numb_mask',`r24')
define(`ulimb',`r25')
define(`rlimb',`r27')

define(`m0a',`r0')
define(`m0b',`r1')
define(`m1a',`r2')
define(`m1b',`r3')
define(`m2a',`r20')
define(`m2b',`r21')
define(`m3a',`r12')
define(`m3b',`r13')

define(`acc0',`r4')
define(`acc1',`r5')
define(`acc2',`r22')
define(`acc3',`r14')

define(`v0',`r6')
define(`v1',`r7')
define(`v2',`r23')
define(`v3',`r15')

C Used for temps: r8 r19 r28

define(`NAIL_BITS',`GMP_NAIL_BITS')
define(`NUMB_BITS',`GMP_NUMB_BITS')

C  This declaration is munged by configure
NAILS_SUPPORT(4-63)

ASM_START()
PROLOGUE(mpn_addmul_4)
	lda	r30,	-240(r30)
	stq	r12,	32(r30)
	stq	r13,	40(r30)
	stq	r14,	48(r30)
	stq	r15,	56(r30)

	lda	numb_mask,-1(r31)
	srl	numb_mask,NAIL_BITS,numb_mask

	ldq	v0,	0(vp)
	ldq	v1,	8(vp)
	ldq	v2,	16(vp)
	ldq	v3,	24(vp)

	bis	r31,	r31,	acc0		C	zero acc0
	sll	v0,NAIL_BITS,	v0
	bis	r31,	r31,	acc1		C	zero acc1
	sll	v1,NAIL_BITS,	v1
	bis	r31,	r31,	acc2		C	zero acc2
	sll	v2,NAIL_BITS,	v2
	bis	r31,	r31,	acc3		C	zero acc3
	sll	v3,NAIL_BITS,	v3
	bis	r31,	r31,	r19

	ldq	ulimb,	0(up)
	lda	up,	8(up)
	mulq	v0,	ulimb,	m0a		C U1
	umulh	v0,	ulimb,	m0b		C U1
	mulq	v1,	ulimb,	m1a		C U1
	umulh	v1,	ulimb,	m1b		C U1
	lda	n,	-1(n)
	mulq	v2,	ulimb,	m2a		C U1
	umulh	v2,	ulimb,	m2b		C U1
	mulq	v3,	ulimb,	m3a		C U1
	umulh	v3,	ulimb,	m3b		C U1
	beq	n,	L(end)			C U0

	ALIGN(16)
L(top):	bis	r31,	r31,	r31		C U1	nop
	ldq	rlimb,	0(rp)			C L0
	ldq	ulimb,	0(up)			C L1
	addq	r19,	acc0,	acc0		C U0	propagate nail

	bis	r31,	r31,	r31		C L0	nop
	bis	r31,	r31,	r31		C U1	nop
	bis	r31,	r31,	r31		C L1	nop
	bis	r31,	r31,	r31		C U0	nop

	lda	rp,	8(rp)			C L0
	srl	m0a,NAIL_BITS,	r8		C U0
	lda	up,	8(up)			C L1
	mulq	v0,	ulimb,	m0a		C U1

	addq	r8,	acc0,	r19		C U0
	addq	m0b,	acc1,	acc0		C L0
	umulh	v0,	ulimb,	m0b		C U1
	bis	r31,	r31,	r31		C L1	nop

	addq	rlimb,	r19,	r19		C L0
	srl	m1a,NAIL_BITS,	r8		C U0
	bis	r31,	r31,	r31		C L1	nop
	mulq	v1,	ulimb,	m1a		C U1

	addq	r8,	acc0,	acc0		C U0
	addq	m1b,	acc2,	acc1		C L0
	umulh	v1,	ulimb,	m1b		C U1
	and	r19,numb_mask,	r28		C L1	extract numb part

	bis	r31,	r31,	r31		C L0	nop
	srl	m2a,NAIL_BITS,	r8		C U0
	lda	n,	-1(n)			C L1
	mulq	v2,	ulimb,	m2a		C U1

	addq	r8,	acc1,	acc1		C L1
	addq	m2b,	acc3,	acc2		C L0
	umulh	v2,	ulimb,	m2b		C U1
	srl	r19,NUMB_BITS,	r19		C U0	extract nail part

	bis	r31,	r31,	r31		C L0	nop
	srl	m3a,NAIL_BITS,	r8		C U0
	stq	r28,	-8(rp)			C L1
	mulq	v3,	ulimb,	m3a		C U1

	addq	r8,	acc2,	acc2		C L0
	bis	r31,	m3b,	acc3		C L1
	umulh	v3,	ulimb,	m3b		C U1
	bne	n,	L(top)			C U0

L(end):	ldq	rlimb,	0(rp)
	addq	r19,	acc0,	acc0		C	propagate nail
	lda	rp,	8(rp)			C FIXME: DELETE
	srl	m0a,NAIL_BITS,	r8		C U0
	addq	r8,	acc0,	r19
	addq	m0b,	acc1,	acc0
	addq	rlimb,	r19,	r19
	srl	m1a,NAIL_BITS,	r8		C U0
	addq	r8,	acc0,	acc0
	addq	m1b,	acc2,	acc1
	and	r19,numb_mask,	r28		C extract limb
	srl	m2a,NAIL_BITS,	r8		C U0
	addq	r8,	acc1,	acc1
	addq	m2b,	acc3,	acc2
	srl	r19,NUMB_BITS,	r19		C extract nail
	srl	m3a,NAIL_BITS,	r8		C U0
	stq	r28,	-8(rp)
	addq	r8,	acc2,	acc2
	bis	r31,	m3b,	acc3

	addq	r19,	acc0,	acc0		C propagate nail
	and	acc0,numb_mask,	r28
	stq	r28,	0(rp)
	srl	acc0,NUMB_BITS,	r19
	addq	r19,	acc1,	acc1

	and	acc1,numb_mask,	r28
	stq	r28,	8(rp)
	srl	acc1,NUMB_BITS,	r19
	addq	r19,	acc2,	acc2

	and	acc2,numb_mask,	r28
	stq	r28,	16(rp)
	srl	acc2,NUMB_BITS,	r19
	addq	r19,	acc3,	r0

	ldq	r12,	32(r30)
	ldq	r13,	40(r30)
	ldq	r14,	48(r30)
	ldq	r15,	56(r30)
	lda	r30,	240(r30)
	ret	r31,	(r26),	1
EPILOGUE()
ASM_END()
