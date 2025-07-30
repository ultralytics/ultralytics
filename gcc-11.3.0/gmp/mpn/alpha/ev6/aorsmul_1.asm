dnl  Alpha ev6 mpn_addmul_1 and mpn_submul_1.

dnl  Copyright 2000, 2003-2005, 2008 Free Software Foundation, Inc.

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
C EV6:     3.5

C  INPUT PARAMETERS
define(`rp',	`r16')
define(`up',	`r17')
define(`n',	`r18')
define(`v0',	`r19')

dnl  This code was written in cooperation with ev6 pipeline expert Steve Root.

dnl  The stores can issue a cycle late so we have paired no-op's to 'catch'
dnl  them, so that further disturbance to the schedule is damped.

dnl  We couldn't pair the loads, because the entangled schedule of the carry's
dnl  has to happen on one side {0} of the machine.

dnl  This is a great schedule for the d_cache, a poor schedule for the b_cache.
dnl  The lockup on U0 means that any stall can't be recovered from.  Consider a
dnl  ldq in L1, say that load gets stalled because it collides with a fill from
dnl  the b_cache.  On the next cycle, this load gets priority.  If first looks
dnl  at L0, and goes there.  The instruction we intended for L0 gets to look at
dnl  L1, which is NOT where we want it.  It either stalls 1, because it can't
dnl  go in L0, or goes there, and causes a further instruction to stall.

dnl  So for b_cache, we're likely going to want to put one or more cycles back
dnl  into the code! And, of course, put in lds prefetch for the rp[] operand.
dnl  At a place where we have an mt followed by a bookkeeping, put the
dnl  bookkeeping in upper, and the prefetch into lower.

dnl  Note, the ldq's and stq's are at the end of the quadpacks.  Note, we'd
dnl  like not to have an ldq or an stq to preceded a conditional branch in a
dnl  quadpack.  The conditional branch moves the retire pointer one cycle
dnl  later.

ifdef(`OPERATION_addmul_1',`
    define(`ADDSUB',	`addq')
    define(`CMPCY',	`cmpult	$2,$1')
    define(`func',	`mpn_addmul_1')
')
ifdef(`OPERATION_submul_1',`
    define(`ADDSUB',	`subq')
    define(`CMPCY',	`cmpult	$1,$2')
    define(`func',	`mpn_submul_1')
')

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_submul_1)

ASM_START()
PROLOGUE(func)
	ldq	r3,	0(up)		C
	and	r18,	7,	r20	C
	lda	r18,	-9(r18)		C
	cmpeq	r20,	1,	r21	C
	beq	r21,	$L1		C

$1mod8:	ldq	r5,	0(rp)		C
	mulq	v0,	r3,	r7	C
	umulh	v0,	r3,	r8	C
	ADDSUB	r5,	r7,	r23	C
	CMPCY(	r5,	r23),	r20	C
	addq	r8,	r20,	r0	C
	stq	r23,	0(rp)		C
	bge	r18,	$ent1		C
	ret	r31,	(r26),	1	C

$L1:	lda	r8,	0(r31)		C zero carry reg
	lda	r24,	0(r31)		C zero carry reg
	cmpeq	r20,	2,	r21	C
	bne	r21,	$2mod8		C
	cmpeq	r20,	3,	r21	C
	bne	r21,	$3mod8		C
	cmpeq	r20,	4,	r21	C
	bne	r21,	$4mod8		C
	cmpeq	r20,	5,	r21	C
	bne	r21,	$5mod8		C
	cmpeq	r20,	6,	r21	C
	bne	r21,	$6mod8		C
	cmpeq	r20,	7,	r21	C
	beq	r21,	$0mod8		C

$7mod8:	ldq	r5,	0(rp)		C
	lda	up,	8(up)		C
	mulq	v0,	r3,	r7	C
	umulh	v0,	r3,	r24	C
	ADDSUB	r5,	r7,	r23	C
	CMPCY(	r5,	r23),	r20	C
	addq	r24,	r20,	r24	C
	stq	r23,	0(rp)		C
	lda	rp,	8(rp)		C
	ldq	r3,	0(up)		C
$6mod8:	ldq	r1,	8(up)		C
	mulq	v0,	r3,	r25	C
	umulh	v0,	r3,	r3	C
	mulq	v0,	r1,	r28	C
	ldq	r0,	16(up)		C
	ldq	r4,	0(rp)		C
	umulh	v0,	r1,	r8	C
	ldq	r1,	24(up)		C
	lda	up,	48(up)		C L1 bookkeeping
	mulq	v0,	r0,	r2	C
	ldq	r5,	8(rp)		C
	lda	rp,	-32(rp)		C L1 bookkeeping
	umulh	v0,	r0,	r6	C
	ADDSUB	r4,	r25,	r25	C lo + acc
	mulq	v0,	r1,	r7	C
	br	r31,	$ent6		C

$ent1:	lda	up,	8(up)		C
	lda	rp,	8(rp)		C
	lda	r8,	0(r0)		C
	ldq	r3,	0(up)		C
$0mod8:	ldq	r1,	8(up)		C
	mulq	v0,	r3,	r2	C
	umulh	v0,	r3,	r6	C
	mulq	v0,	r1,	r7	C
	ldq	r0,	16(up)		C
	ldq	r4,	0(rp)		C
	umulh	v0,	r1,	r24	C
	ldq	r1,	24(up)		C
	mulq	v0,	r0,	r25	C
	ldq	r5,	8(rp)		C
	umulh	v0,	r0,	r3	C
	ADDSUB	r4,	r2,	r2	C lo + acc
	mulq	v0,	r1,	r28	C
	lda	rp,	-16(rp)		C
	br	r31,	$ent0		C

$3mod8:	ldq	r5,	0(rp)		C
	lda	up,	8(up)		C
	mulq	v0,	r3,	r7	C
	umulh	v0,	r3,	r8	C
	ADDSUB	r5,	r7,	r23	C
	CMPCY(	r5,	r23),	r20	C
	addq	r8,	r20,	r24	C
	stq	r23,	0(rp)		C
	lda	rp,	8(rp)		C
	ldq	r3,	0(up)		C
$2mod8:	ldq	r1,	8(up)		C
	mulq	v0,	r3,	r25	C
	umulh	v0,	r3,	r3	C
	mulq	v0,	r1,	r28	C
	ble	r18,	$n23		C
	ldq	r0,	16(up)		C
	ldq	r4,	0(rp)		C
	umulh	v0,	r1,	r8	C
	ldq	r1,	24(up)		C
	lda	up,	16(up)		C L1 bookkeeping
	mulq	v0,	r0,	r2	C
	ldq	r5,	8(rp)		C
	lda	rp,	0(rp)		C L1 bookkeeping
	umulh	v0,	r0,	r6	C
	ADDSUB	r4,	r25,	r25	C lo + acc
	mulq	v0,	r1,	r7	C
	br	r31,	$ent2		C

$5mod8:	ldq	r5,	0(rp)		C
	lda	up,	8(up)		C
	mulq	v0,	r3,	r7	C
	umulh	v0,	r3,	r24	C
	ADDSUB	r5,	r7,	r23	C
	CMPCY(	r5,	r23),	r20	C
	addq	r24,	r20,	r8	C
	stq	r23,	0(rp)		C
	lda	rp,	8(rp)		C
	ldq	r3,	0(up)		C
$4mod8:	ldq	r1,	8(up)		C
	mulq	v0,	r3,	r2	C
	umulh	v0,	r3,	r6	C
	mulq	v0,	r1,	r7	C
	ldq	r0,	16(up)		C
	ldq	r4,	0(rp)		C
	umulh	v0,	r1,	r24	C
	ldq	r1,	24(up)		C
	lda	up,	32(up)		C L1 bookkeeping
	mulq	v0,	r0,	r25	C
	ldq	r5,	8(rp)		C
	lda	rp,	16(rp)		C L1 bookkeeping
	umulh	v0,	r0,	r3	C
	ADDSUB	r4,	r2,	r2	C lo + acc
	mulq	v0,	r1,	r28	C
	CMPCY(	r4,	r2),	r20	C L0 lo add => carry
	ADDSUB	r2,	r8,	r22	C U0 hi add => answer
	ble	r18,	$Lend		C
	ALIGN(16)
$Loop:
	bis	r31,	r31,	r31	C U1 mt
	CMPCY(	r2,	r22),	r21	C L0 hi add => carry
	addq	r6,	r20,	r6	C U0 hi mul + carry
	ldq	r0,	0(up)		C

	bis	r31,	r31,	r31	C U1 mt
	ADDSUB	r5,	r7,	r7	C L0 lo + acc
	addq	r6,	r21,	r6	C U0 hi mul + carry
	ldq	r4,	0(rp)		C L1

	umulh	v0,	r1,	r8	C U1
	CMPCY(	r5,	r7),	r20	C L0 lo add => carry
	ADDSUB	r7,	r6,	r23	C U0 hi add => answer
	ldq	r1,	8(up)		C L1

	mulq	v0,	r0,	r2	C U1
	CMPCY(	r7,	r23),	r21	C L0 hi add => carry
	addq	r24,	r20,	r24	C U0 hi mul + carry
	ldq	r5,	8(rp)		C L1

	umulh	v0,	r0,	r6	C U1
	ADDSUB	r4,	r25,	r25	C U0 lo + acc
	stq	r22,	-16(rp)		C L0
	stq	r23,	-8(rp)		C L1

	bis	r31,	r31,	r31	C L0 st slosh
	mulq	v0,	r1,	r7	C U1
	bis	r31,	r31,	r31	C L1 st slosh
	addq	r24,	r21,	r24	C U0 hi mul + carry
$ent2:
	CMPCY(	r4,	r25),	r20	C L0 lo add => carry
	bis	r31,	r31,	r31	C U1 mt
	lda	r18,	-8(r18)		C L1 bookkeeping
	ADDSUB	r25,	r24,	r22	C U0 hi add => answer

	bis	r31,	r31,	r31	C U1 mt
	CMPCY(	r25,	r22),	r21	C L0 hi add => carry
	addq	r3,	r20,	r3	C U0 hi mul + carry
	ldq	r0,	16(up)		C L1

	bis	r31,	r31,	r31	C U1 mt
	ADDSUB	r5,	r28,	r28	C L0 lo + acc
	addq	r3,	r21,	r3	C U0 hi mul + carry
	ldq	r4,	16(rp)		C L1

	umulh	v0,	r1,	r24	C U1
	CMPCY(	r5,	r28),	r20	C L0 lo add => carry
	ADDSUB	r28,	r3,	r23	C U0 hi add => answer
	ldq	r1,	24(up)		C L1

	mulq	v0,	r0,	r25	C U1
	CMPCY(	r28,	r23),	r21	C L0 hi add => carry
	addq	r8,	r20,	r8	C U0 hi mul + carry
	ldq	r5,	24(rp)		C L1

	umulh	v0,	r0,	r3	C U1
	ADDSUB	r4,	r2,	r2	C U0 lo + acc
	stq	r22,	0(rp)		C L0
	stq	r23,	8(rp)		C L1

	bis	r31,	r31,	r31	C L0 st slosh
	mulq	v0,	r1,	r28	C U1
	bis	r31,	r31,	r31	C L1 st slosh
	addq	r8,	r21,	r8	C U0 hi mul + carry
$ent0:
	CMPCY(	r4,	r2),	r20	C L0 lo add => carry
	bis	r31,	r31,	r31	C U1 mt
	lda	up,	64(up)		C L1 bookkeeping
	ADDSUB	r2,	r8,	r22	C U0 hi add => answer

	bis	r31,	r31,	r31	C U1 mt
	CMPCY(	r2,	r22),	r21	C L0 hi add => carry
	addq	r6,	r20,	r6	C U0 hi mul + carry
	ldq	r0,	-32(up)		C L1

	bis	r31,	r31,	r31	C U1 mt
	ADDSUB	r5,	r7,	r7	C L0 lo + acc
	addq	r6,	r21,	r6	C U0 hi mul + carry
	ldq	r4,	32(rp)		C L1

	umulh	v0,	r1,	r8	C U1
	CMPCY(	r5,	r7),	r20	C L0 lo add => carry
	ADDSUB	r7,	r6,	r23	C U0 hi add => answer
	ldq	r1,	-24(up)		C L1

	mulq	v0,	r0,	r2	C U1
	CMPCY(	r7,	r23),	r21	C L0 hi add => carry
	addq	r24,	r20,	r24	C U0 hi mul + carry
	ldq	r5,	40(rp)		C L1

	umulh	v0,	r0,	r6	C U1
	ADDSUB	r4,	r25,	r25	C U0 lo + acc
	stq	r22,	16(rp)		C L0
	stq	r23,	24(rp)		C L1

	bis	r31,	r31,	r31	C L0 st slosh
	mulq	v0,	r1,	r7	C U1
	bis	r31,	r31,	r31	C L1 st slosh
	addq	r24,	r21,	r24	C U0 hi mul + carry
$ent6:
	CMPCY(	r4,	r25),	r20	C L0 lo add => carry
	bis	r31,	r31,	r31	C U1 mt
	lda	rp,	64(rp)		C L1 bookkeeping
	ADDSUB	r25,	r24,	r22	C U0 hi add => answer

	bis	r31,	r31,	r31	C U1 mt
	CMPCY(	r25,	r22),	r21	C L0 hi add => carry
	addq	r3,	r20,	r3	C U0 hi mul + carry
	ldq	r0,	-16(up)		C L1

	bis	r31,	r31,	r31	C U1 mt
	ADDSUB	r5,	r28,	r28	C L0 lo + acc
	addq	r3,	r21,	r3	C U0 hi mul + carry
	ldq	r4,	-16(rp)		C L1

	umulh	v0,	r1,	r24	C U1
	CMPCY(	r5,	r28),	r20	C L0 lo add => carry
	ADDSUB	r28,	r3,	r23	C U0 hi add => answer
	ldq	r1,	-8(up)		C L1

	mulq	v0,	r0,	r25	C U1
	CMPCY(	r28,	r23),	r21	C L0 hi add => carry
	addq	r8,	r20,	r8	C U0 hi mul + carry
	ldq	r5,	-8(rp)		C L1

	umulh	v0,	r0,	r3	C U1
	ADDSUB	r4,	r2,	r2	C U0 lo + acc
	stq	r22,	-32(rp)		C L0
	stq	r23,	-24(rp)		C L1

	bis	r31,	r31,	r31	C L0 st slosh
	mulq	v0,	r1,	r28	C U1
	bis	r31,	r31,	r31	C L1 st slosh
	addq	r8,	r21,	r8	C U0 hi mul + carry

	CMPCY(	r4,	r2),	r20	C L0 lo add => carry
	ADDSUB	r2,	r8,	r22	C U0 hi add => answer
	ldl	r31,	256(up)		C prefetch up[]
	bgt	r18,	$Loop		C U1 bookkeeping

$Lend:	CMPCY(	r2,	r22),	r21	C
	addq	r6,	r20,	r6	C
	ADDSUB	r5,	r7,	r7	C
	addq	r6,	r21,	r6	C
	ldq	r4,	0(rp)		C
	umulh	v0,	r1,	r8	C
	CMPCY(	r5,	r7),	r20	C
	ADDSUB	r7,	r6,	r23	C
	CMPCY(r7,	r23),	r21	C
	addq	r24,	r20,	r24	C
	ldq	r5,	8(rp)		C
	ADDSUB	r4,	r25,	r25	C
	stq	r22,	-16(rp)		C
	stq	r23,	-8(rp)		C
	addq	r24,	r21,	r24	C
	br	L(x)

	ALIGN(16)
$n23:	ldq	r4,	0(rp)		C
	ldq	r5,	8(rp)		C
	umulh	v0,	r1,	r8	C
	ADDSUB	r4,	r25,	r25	C
L(x):	CMPCY(	r4,	r25),	r20	C
	ADDSUB	r25,	r24,	r22	C
	CMPCY(	r25,	r22),	r21	C
	addq	r3,	r20,	r3	C
	ADDSUB	r5,	r28,	r28	C
	addq	r3,	r21,	r3	C
	CMPCY(	r5,	r28),	r20	C
	ADDSUB	r28,	r3,	r23	C
	CMPCY(	r28,	r23),	r21	C
	addq	r8,	r20,	r8	C
	stq	r22,	0(rp)		C
	stq	r23,	8(rp)		C
	addq	r8,	r21,	r0	C
	ret	r31,	(r26),	1	C
EPILOGUE()
ASM_END()
