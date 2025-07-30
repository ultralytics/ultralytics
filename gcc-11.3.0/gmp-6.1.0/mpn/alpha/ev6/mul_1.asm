dnl  Alpha ev6 mpn_mul_1 -- Multiply a limb vector with a limb and store the
dnl  result in a second limb vector.

dnl  Copyright 2000, 2001, 2005 Free Software Foundation, Inc.

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

C INPUT PARAMETERS
C res_ptr	r16
C s1_ptr	r17
C size		r18
C s2_limb	r19

C This code runs at 2.25 cycles/limb on EV6.

C This code was written in close cooperation with ev6 pipeline expert
C Steve Root.  Any errors are tege's fault, though.

C Code structure:

C  code for n < 8
C  code for n > 8	code for (n mod 8)
C			code for (n div 8)	feed-in code
C						8-way unrolled loop
C						wind-down code

C Some notes about unrolled loop:
C
C   r1-r8     multiplies and workup
C   r21-r28   multiplies and workup
C   r9-r12    loads
C   r0       -1
C   r20,r29,r13-r15  scramble
C
C   We're doing 7 of the 8 carry propagations with a br fixup code and 1 with a
C   put-the-carry-into-hi.  The idea is that these branches are very rarely
C   taken, and since a non-taken branch consumes no resources, that is better
C   than an addq.
C
C   Software pipeline: a load in cycle #09, feeds a mul in cycle #16, feeds an
C   add NEXT cycle #09 which feeds a store in NEXT cycle #02

C The code could use some further work:
C   1. Speed up really small multiplies.  The default alpha/mul_1.asm code is
C      faster than this for size < 3.
C   2. Improve feed-in code, perhaps with the equivalent of switch(n%8) unless
C      that is too costly.
C   3. Consider using 4-way unrolling, even if that runs slower.
C   4. Reduce register usage.  In particular, try to avoid using r29.

ASM_START()
PROLOGUE(mpn_mul_1)
	cmpult	r18,	8,	r1
	beq	r1,	$Large
$Lsmall:
	ldq	r2,0(r17)	C r2 = s1_limb
	lda	r18,-1(r18)	C size--
	mulq	r2,r19,r3	C r3 = prod_low
	bic	r31,r31,r4	C clear cy_limb
	umulh	r2,r19,r0	C r0 = prod_high
	beq	r18,$Le1a	C jump if size was == 1
	ldq	r2,8(r17)	C r2 = s1_limb
	lda	r18,-1(r18)	C size--
	stq	r3,0(r16)
	beq	r18,$Le2a	C jump if size was == 2
	ALIGN(8)
$Lopa:	mulq	r2,r19,r3	C r3 = prod_low
	addq	r4,r0,r0	C cy_limb = cy_limb + 'cy'
	lda	r18,-1(r18)	C size--
	umulh	r2,r19,r4	C r4 = cy_limb
	ldq	r2,16(r17)	C r2 = s1_limb
	lda	r17,8(r17)	C s1_ptr++
	addq	r3,r0,r3	C r3 = cy_limb + prod_low
	stq	r3,8(r16)
	cmpult	r3,r0,r0	C r0 = carry from (cy_limb + prod_low)
	lda	r16,8(r16)	C res_ptr++
	bne	r18,$Lopa

$Le2a:	mulq	r2,r19,r3	C r3 = prod_low
	addq	r4,r0,r0	C cy_limb = cy_limb + 'cy'
	umulh	r2,r19,r4	C r4 = cy_limb
	addq	r3,r0,r3	C r3 = cy_limb + prod_low
	cmpult	r3,r0,r0	C r0 = carry from (cy_limb + prod_low)
	stq	r3,8(r16)
	addq	r4,r0,r0	C cy_limb = prod_high + cy
	ret	r31,(r26),1
$Le1a:	stq	r3,0(r16)
	ret	r31,(r26),1

$Large:
	lda	r30,	-224(r30)
	stq	r26,	0(r30)
	stq	r9,	8(r30)
	stq	r10,	16(r30)
	stq	r11,	24(r30)
	stq	r12,	32(r30)
	stq	r13,	40(r30)
	stq	r14,	48(r30)
	stq	r15,	56(r30)
	stq	r29,	64(r30)

	and	r18,	7,	r20	C count for the first loop, 0-7
	srl	r18,	3,	r18	C count for unrolled loop
	bis	r31,	r31,	r21
	beq	r20,	$L_8_or_more	C skip first loop

$L_9_or_more:
	ldq	r2,0(r17)	C r2 = s1_limb
	lda	r17,8(r17)	C s1_ptr++
	lda	r20,-1(r20)	C size--
	mulq	r2,r19,r3	C r3 = prod_low
	umulh	r2,r19,r21	C r21 = prod_high
	beq	r20,$Le1b	C jump if size was == 1
	bis	r31, r31, r0	C FIXME: shouldn't need this
	ldq	r2,0(r17)	C r2 = s1_limb
	lda	r17,8(r17)	C s1_ptr++
	lda	r20,-1(r20)	C size--
	stq	r3,0(r16)
	lda	r16,8(r16)	C res_ptr++
	beq	r20,$Le2b	C jump if size was == 2
	ALIGN(8)
$Lopb:	mulq	r2,r19,r3	C r3 = prod_low
	addq	r21,r0,r0	C cy_limb = cy_limb + 'cy'
	lda	r20,-1(r20)	C size--
	umulh	r2,r19,r21	C r21 = prod_high
	ldq	r2,0(r17)	C r2 = s1_limb
	lda	r17,8(r17)	C s1_ptr++
	addq	r3,r0,r3	C r3 = cy_limb + prod_low
	stq	r3,0(r16)
	cmpult	r3,r0,r0	C r0 = carry from (cy_limb + prod_low)
	lda	r16,8(r16)	C res_ptr++
	bne	r20,$Lopb

$Le2b:	mulq	r2,r19,r3	C r3 = prod_low
	addq	r21,r0,r0	C cy_limb = cy_limb + 'cy'
	umulh	r2,r19,r21	C r21 = prod_high
	addq	r3,r0,r3	C r3 = cy_limb + prod_low
	cmpult	r3,r0,r0	C r0 = carry from (cy_limb + prod_low)
	stq	r3,0(r16)
	lda	r16,8(r16)	C res_ptr++
	addq	r21,r0,r21	C cy_limb = prod_high + cy
	br	r31,	$L_8_or_more
$Le1b:	stq	r3,0(r16)
	lda	r16,8(r16)	C res_ptr++

$L_8_or_more:
	lda	r0,	-1(r31)		C put -1 in r0, for tricky loop control
	lda	r17,	-32(r17)	C L1 bookkeeping
	lda	r18,	-1(r18)		C decrement count

	ldq	r9,	32(r17)		C L1
	ldq	r10,	40(r17)		C L1
	mulq	r9,	r19,	r22	C U1 #07
	ldq	r11,	48(r17)		C L1
	umulh	r9,	r19,	r23	C U1 #08
	ldq	r12,	56(r17)		C L1
	mulq	r10,	r19,	r24	C U1 #09
	ldq	r9,	64(r17)		C L1

	lda	r17,	64(r17)		C L1 bookkeeping

	umulh	r10,	r19,	r25	C U1 #11
	mulq	r11,	r19,	r26	C U1 #12
	umulh	r11,	r19,	r27	C U1 #13
	mulq	r12,	r19,	r28	C U1 #14
	ldq	r10,	8(r17)		C L1
	umulh	r12,	r19,	r1	C U1 #15
	ldq	r11,	16(r17)		C L1
	mulq	r9,	r19,	r2	C U1 #16
	ldq	r12,	24(r17)		C L1
	umulh	r9,	r19,	r3	C U1 #17
	addq	r21,	r22,	r13	C L1 mov
	mulq	r10,	r19,	r4	C U1 #18
	addq	r23,	r24,	r22	C L0 sum 2 mul's
	cmpult	r13,	r21,	r14	C L1 carry from sum
	bgt	r18,	$L_16_or_more

	cmpult	r22,	r24,	r24	C U0 carry from sum
	umulh	r10,	r19,	r5	C U1 #02
	addq	r25,	r26,	r23	C U0 sum 2 mul's
	mulq	r11,	r19,	r6	C U1 #03
	cmpult	r23,	r26,	r25	C U0 carry from sum
	umulh	r11,	r19,	r7	C U1 #04
	addq	r27,	r28,	r28	C U0 sum 2 mul's
	mulq	r12,	r19,	r8	C U1 #05
	cmpult	r28,	r27,	r15	C L0 carry from sum
	lda	r16,	32(r16)		C L1 bookkeeping
	addq	r13,	r31,	r13	C U0 start carry cascade
	umulh	r12,	r19,	r21	C U1 #06
	br	r31,	$ret0c

$L_16_or_more:
C ---------------------------------------------------------------
	subq	r18,1,r18
	cmpult	r22,	r24,	r24	C U0 carry from sum
	ldq	r9,	32(r17)		C L1

	umulh	r10,	r19,	r5	C U1 #02
	addq	r25,	r26,	r23	C U0 sum 2 mul's
	mulq	r11,	r19,	r6	C U1 #03
	cmpult	r23,	r26,	r25	C U0 carry from sum
	umulh	r11,	r19,	r7	C U1 #04
	addq	r27,	r28,	r28	C U0 sum 2 mul's
	mulq	r12,	r19,	r8	C U1 #05
	cmpult	r28,	r27,	r15	C L0 carry from sum
	lda	r16,	32(r16)		C L1 bookkeeping
	addq	r13,	r31,	r13	C U0 start carry cascade

	umulh	r12,	r19,	r21	C U1 #06
C	beq	r13,	$fix0w		C U0
$ret0w:	addq	r22,	r14,	r26	C L0
	ldq	r10,	40(r17)		C L1

	mulq	r9,	r19,	r22	C U1 #07
	beq	r26,	$fix1w		C U0
$ret1w:	addq	r23,	r24,	r27	C L0
	ldq	r11,	48(r17)		C L1

	umulh	r9,	r19,	r23	C U1 #08
	beq	r27,	$fix2w		C U0
$ret2w:	addq	r28,	r25,	r28	C L0
	ldq	r12,	56(r17)		C L1

	mulq	r10,	r19,	r24	C U1 #09
	beq	r28,	$fix3w		C U0
$ret3w:	addq	r1,	r2,	r20	C L0 sum 2 mul's
	ldq	r9,	64(r17)		C L1

	addq	r3,	r4,	r2	C L0 #10 2 mul's
	lda	r17,	64(r17)		C L1 bookkeeping
	cmpult	r20,	r1,	r29	C U0 carry from sum

	umulh	r10,	r19,	r25	C U1 #11
	cmpult	r2,	r4,	r4	C U0 carry from sum
	stq	r13,	-32(r16)	C L0
	stq	r26,	-24(r16)	C L1

	mulq	r11,	r19,	r26	C U1 #12
	addq	r5,	r6,	r14	C U0 sum 2 mul's
	stq	r27,	-16(r16)	C L0
	stq	r28,	-8(r16)		C L1

	umulh	r11,	r19,	r27	C U1 #13
	cmpult	r14,	r6,	r3	C U0 carry from sum
C could do cross-jumping here:
C	bra	$L_middle_of_unrolled_loop
	mulq	r12,	r19,	r28	C U1 #14
	addq	r7,	r3,	r5	C L0 eat carry
	addq	r20,	r15,	r20	C U0 carry cascade
	ldq	r10,	8(r17)		C L1

	umulh	r12,	r19,	r1	C U1 #15
	beq	r20,	$fix4		C U0
$ret4w:	addq	r2,	r29,	r6	C L0
	ldq	r11,	16(r17)		C L1

	mulq	r9,	r19,	r2	C U1 #16
	beq	r6,	$fix5		C U0
$ret5w:	addq	r14,	r4,	r7	C L0
	ldq	r12,	24(r17)		C L1

	umulh	r9,	r19,	r3	C U1 #17
	beq	r7,	$fix6		C U0
$ret6w:	addq	r5,	r8,	r8	C L0 sum 2
	addq	r21,	r22,	r13	C L1 sum 2 mul's

	mulq	r10,	r19,	r4	C U1 #18
	addq	r23,	r24,	r22	C L0 sum 2 mul's
	cmpult	r13,	r21,	r14	C L1 carry from sum
	ble	r18,	$Lend		C U0
C ---------------------------------------------------------------
	ALIGN(16)
$Loop:
	umulh	r0,	r18,	r18	C U1 #01 decrement r18!
	cmpult	r8,	r5,	r29	C L0 carry from last bunch
	cmpult	r22,	r24,	r24	C U0 carry from sum
	ldq	r9,	32(r17)		C L1

	umulh	r10,	r19,	r5	C U1 #02
	addq	r25,	r26,	r23	C U0 sum 2 mul's
	stq	r20,	0(r16)		C L0
	stq	r6,	8(r16)		C L1

	mulq	r11,	r19,	r6	C U1 #03
	cmpult	r23,	r26,	r25	C U0 carry from sum
	stq	r7,	16(r16)		C L0
	stq	r8,	24(r16)		C L1

	umulh	r11,	r19,	r7	C U1 #04
	bis	r31,	r31,	r31	C L0 st slosh
	bis	r31,	r31,	r31	C L1 st slosh
	addq	r27,	r28,	r28	C U0 sum 2 mul's

	mulq	r12,	r19,	r8	C U1 #05
	cmpult	r28,	r27,	r15	C L0 carry from sum
	lda	r16,	64(r16)		C L1 bookkeeping
	addq	r13,	r29,	r13	C U0 start carry cascade

	umulh	r12,	r19,	r21	C U1 #06
	beq	r13,	$fix0		C U0
$ret0:	addq	r22,	r14,	r26	C L0
	ldq	r10,	40(r17)		C L1

	mulq	r9,	r19,	r22	C U1 #07
	beq	r26,	$fix1		C U0
$ret1:	addq	r23,	r24,	r27	C L0
	ldq	r11,	48(r17)		C L1

	umulh	r9,	r19,	r23	C U1 #08
	beq	r27,	$fix2		C U0
$ret2:	addq	r28,	r25,	r28	C L0
	ldq	r12,	56(r17)		C L1

	mulq	r10,	r19,	r24	C U1 #09
	beq	r28,	$fix3		C U0
$ret3:	addq	r1,	r2,	r20	C L0 sum 2 mul's
	ldq	r9,	64(r17)		C L1

	addq	r3,	r4,	r2	C L0 #10 2 mul's
	bis	r31,	r31,	r31	C U1 mul hole
	lda	r17,	64(r17)		C L1 bookkeeping
	cmpult	r20,	r1,	r29	C U0 carry from sum

	umulh	r10,	r19,	r25	C U1 #11
	cmpult	r2,	r4,	r4	C U0 carry from sum
	stq	r13,	-32(r16)	C L0
	stq	r26,	-24(r16)	C L1

	mulq	r11,	r19,	r26	C U1 #12
	addq	r5,	r6,	r14	C U0 sum 2 mul's
	stq	r27,	-16(r16)	C L0
	stq	r28,	-8(r16)		C L1

	umulh	r11,	r19,	r27	C U1 #13
	bis	r31,	r31,	r31	C L0 st slosh
	bis	r31,	r31,	r31	C L1 st slosh
	cmpult	r14,	r6,	r3	C U0 carry from sum
$L_middle_of_unrolled_loop:
	mulq	r12,	r19,	r28	C U1 #14
	addq	r7,	r3,	r5	C L0 eat carry
	addq	r20,	r15,	r20	C U0 carry cascade
	ldq	r10,	8(r17)		C L1

	umulh	r12,	r19,	r1	C U1 #15
	beq	r20,	$fix4		C U0
$ret4:	addq	r2,	r29,	r6	C L0
	ldq	r11,	16(r17)		C L1

	mulq	r9,	r19,	r2	C U1 #16
	beq	r6,	$fix5		C U0
$ret5:	addq	r14,	r4,	r7	C L0
	ldq	r12,	24(r17)		C L1

	umulh	r9,	r19,	r3	C U1 #17
	beq	r7,	$fix6		C U0
$ret6:	addq	r5,	r8,	r8	C L0 sum 2
	addq	r21,	r22,	r13	C L1 sum 2 mul's

	mulq	r10,	r19,	r4	C U1 #18
	addq	r23,	r24,	r22	C L0 sum 2 mul's
	cmpult	r13,	r21,	r14	C L1 carry from sum
	bgt	r18,	$Loop		C U0
C ---------------------------------------------------------------
$Lend:
	cmpult	r8,	r5,	r29	C L0 carry from last bunch
	cmpult	r22,	r24,	r24	C U0 carry from sum

	umulh	r10,	r19,	r5	C U1 #02
	addq	r25,	r26,	r23	C U0 sum 2 mul's
	stq	r20,	0(r16)		C L0
	stq	r6,	8(r16)		C L1

	mulq	r11,	r19,	r6	C U1 #03
	cmpult	r23,	r26,	r25	C U0 carry from sum
	stq	r7,	16(r16)		C L0
	stq	r8,	24(r16)		C L1

	umulh	r11,	r19,	r7	C U1 #04
	addq	r27,	r28,	r28	C U0 sum 2 mul's

	mulq	r12,	r19,	r8	C U1 #05
	cmpult	r28,	r27,	r15	C L0 carry from sum
	lda	r16,	64(r16)		C L1 bookkeeping
	addq	r13,	r29,	r13	C U0 start carry cascade

	umulh	r12,	r19,	r21	C U1 #06
	beq	r13,	$fix0c		C U0
$ret0c:	addq	r22,	r14,	r26	C L0
	beq	r26,	$fix1c		C U0
$ret1c:	addq	r23,	r24,	r27	C L0
	beq	r27,	$fix2c		C U0
$ret2c:	addq	r28,	r25,	r28	C L0
	beq	r28,	$fix3c		C U0
$ret3c:	addq	r1,	r2,	r20	C L0 sum 2 mul's
	addq	r3,	r4,	r2	C L0 #10 2 mul's
	lda	r17,	64(r17)		C L1 bookkeeping
	cmpult	r20,	r1,	r29	C U0 carry from sum
	cmpult	r2,	r4,	r4	C U0 carry from sum
	stq	r13,	-32(r16)	C L0
	stq	r26,	-24(r16)	C L1
	addq	r5,	r6,	r14	C U0 sum 2 mul's
	stq	r27,	-16(r16)	C L0
	stq	r28,	-8(r16)		C L1
	cmpult	r14,	r6,	r3	C U0 carry from sum
	addq	r7,	r3,	r5	C L0 eat carry
	addq	r20,	r15,	r20	C U0 carry cascade
	beq	r20,	$fix4c		C U0
$ret4c:	addq	r2,	r29,	r6	C L0
	beq	r6,	$fix5c		C U0
$ret5c:	addq	r14,	r4,	r7	C L0
	beq	r7,	$fix6c		C U0
$ret6c:	addq	r5,	r8,	r8	C L0 sum 2
	cmpult	r8,	r5,	r29	C L0 carry from last bunch
	stq	r20,	0(r16)		C L0
	stq	r6,	8(r16)		C L1
	stq	r7,	16(r16)		C L0
	stq	r8,	24(r16)		C L1
	addq	r29,	r21,	r0

	ldq	r26,	0(r30)
	ldq	r9,	8(r30)
	ldq	r10,	16(r30)
	ldq	r11,	24(r30)
	ldq	r12,	32(r30)
	ldq	r13,	40(r30)
	ldq	r14,	48(r30)
	ldq	r15,	56(r30)
	ldq	r29,	64(r30)
	lda	r30,	224(r30)
	ret	r31,	(r26),	1

C $fix0w:	bis	r14,	r29,	r14	C join carries
C	br	r31,	$ret0w
$fix1w:	bis	r24,	r14,	r24	C join carries
	br	r31,	$ret1w
$fix2w:	bis	r25,	r24,	r25	C join carries
	br	r31,	$ret2w
$fix3w:	bis	r15,	r25,	r15	C join carries
	br	r31,	$ret3w
$fix0:	bis	r14,	r29,	r14	C join carries
	br	r31,	$ret0
$fix1:	bis	r24,	r14,	r24	C join carries
	br	r31,	$ret1
$fix2:	bis	r25,	r24,	r25	C join carries
	br	r31,	$ret2
$fix3:	bis	r15,	r25,	r15	C join carries
	br	r31,	$ret3
$fix4:	bis	r29,	r15,	r29	C join carries
	br	r31,	$ret4
$fix5:	bis	r4,	r29,	r4	C join carries
	br	r31,	$ret5
$fix6:	addq	r5,	r4,	r5	C can't carry twice!
	br	r31,	$ret6
$fix0c:	bis	r14,	r29,	r14	C join carries
	br	r31,	$ret0c
$fix1c:	bis	r24,	r14,	r24	C join carries
	br	r31,	$ret1c
$fix2c:	bis	r25,	r24,	r25	C join carries
	br	r31,	$ret2c
$fix3c:	bis	r15,	r25,	r15	C join carries
	br	r31,	$ret3c
$fix4c:	bis	r29,	r15,	r29	C join carries
	br	r31,	$ret4c
$fix5c:	bis	r4,	r29,	r4	C join carries
	br	r31,	$ret5c
$fix6c:	addq	r5,	r4,	r5	C can't carry twice!
	br	r31,	$ret6c

EPILOGUE(mpn_mul_1)
ASM_END()
