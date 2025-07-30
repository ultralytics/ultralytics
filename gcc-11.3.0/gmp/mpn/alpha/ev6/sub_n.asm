dnl  Alpha ev6 mpn_sub_n -- Subtract two limb vectors of the same length > 0
dnl  and store difference in a third limb vector.

dnl  Copyright 2000, 2003, 2005 Free Software Foundation, Inc.

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
C EV4:     ?
C EV5:     5.4
C EV6:     2.125

C  INPUT PARAMETERS
C  rp	r16
C  up	r17
C  vp	r18
C  n	r19
C  cy	r20   (for mpn_add_nc)

C TODO
C   Finish cleaning up cy registers r22, r23 (make them use cy0/cy1)
C   Use multi-pronged feed-in.
C   Perform additional micro-tuning

C  This code was written in cooperation with ev6 pipeline expert Steve Root.

C  Pair loads and stores where possible
C  Store pairs oct-aligned where possible (didn't need it here)
C  Stores are delayed every third cycle
C  Loads and stores are delayed by fills
C  U stays still, put code there where possible (note alternation of U1 and U0)
C  L moves because of loads and stores
C  Note dampers in L to limit damage

C  This odd-looking optimization expects that were having random bits in our
C  data, so that a pure zero result is unlikely. so we penalize the unlikely
C  case to help the common case.

define(`u0', `r0')  define(`u1', `r3')
define(`v0', `r1')  define(`v1', `r4')

define(`cy0', `r20')  define(`cy1', `r21')

MULFUNC_PROLOGUE(mpn_sub_n mpn_sub_nc)

ASM_START()
PROLOGUE(mpn_sub_nc)
	br	r31,	$entry
EPILOGUE()
PROLOGUE(mpn_sub_n)
	bis	r31,	r31,	cy0	C clear carry in
$entry:	cmpult	r19,	5,	r22	C L1 move counter
	ldq	u1,	0(r17)		C L0 get next ones
	ldq	v1,	0(r18)		C L1
	bne	r22,	$Lsmall

	ldq	u0,	8(r17)		C L0 get next ones
	ldq	v0,	8(r18)		C L1
	subq	u1,	v1,	r5	C U0 sub two data

	cmpult	u1,	v1,	r23	C U0 did it borrow
	ldq	u1,	16(r17)		C L0 get next ones
	ldq	v1,	16(r18)		C L1

	subq	u0,	v0,	r8	C U1 sub two data
	subq	r5,	cy0,	r24	C U0 borrow in

	cmpult	u0,	v0,	r22	C U1 did it borrow
	beq	r5,	$fix5f		C U0 fix exact zero
$ret5f:	ldq	u0,	24(r17)		C L0 get next ones
	ldq	v0,	24(r18)		C L1

	subq	r8,	r23,	r25	C U1 borrow from last
	subq	u1,	v1,	r7	C U0 sub two data

	beq	r8,	$fix6f		C U1 fix exact zero
$ret6f:	cmpult	u1,	v1,	r23	C U0 did it borrow
	ldq	u1,	32(r17)		C L0 get next ones
	ldq	v1,	32(r18)		C L1

	lda	r17,	40(r17)		C L0 move pointer
	lda	r18,	40(r18)		C L1 move pointer

	lda	r16,	-8(r16)
	lda	r19,	-13(r19)	C L1 move counter
	blt	r19,	$Lend		C U1 loop control


C Main loop.  8-way unrolled.
	ALIGN(16)
$Loop:	subq	u0,	v0,	r2	C U1 sub two data
	stq	r24,	8(r16)		C L0 put an answer
	subq	r7,	r22,	r24	C U0 borrow from last
	stq	r25,	16(r16)		C L1 pair

	cmpult	u0,	v0,	cy1	C U1 did it borrow
	beq	r7,	$fix7		C U0 fix exact 0
$ret7:	ldq	u0,	0(r17)		C L0 get next ones
	ldq	v0,	0(r18)		C L1

	bis	r31,	r31,	r31	C L  damp out
	subq	r2,	r23,	r25	C U1 borrow from last
	bis	r31,	r31,	r31	C L  moves in L !
	subq	u1,	v1,	r5	C U0 sub two data

	beq	r2,	$fix0		C U1 fix exact zero
$ret0:	cmpult	u1,	v1,	cy0	C U0 did it borrow
	ldq	u1,	8(r17)		C L0 get next ones
	ldq	v1,	8(r18)		C L1

	subq	u0,	v0,	r8	C U1 sub two data
	stq	r24,	24(r16)		C L0 store pair
	subq	r5,	cy1,	r24	C U0 borrow from last
	stq	r25,	32(r16)		C L1

	cmpult	u0,	v0,	r22	C U1 did it borrow
	beq	r5,	$fix1		C U0 fix exact zero
$ret1:	ldq	u0,	16(r17)		C L0 get next ones
	ldq	v0,	16(r18)		C L1

	lda	r16,	64(r16)		C L0 move pointer
	subq	r8,	cy0,	r25	C U1 borrow from last
	lda	r19,	-8(r19)		C L1 move counter
	subq	u1,	v1,	r7	C U0 sub two data

	beq	r8,	$fix2		C U1 fix exact zero
$ret2:	cmpult	u1,	v1,	r23	C U0 did it borrow
	ldq	u1,	24(r17)		C L0 get next ones
	ldq	v1,	24(r18)		C L1

	subq	u0,	v0,	r2	C U1 sub two data
	stq	r24,	-24(r16)	C L0 put an answer
	subq	r7,	r22,	r24	C U0 borrow from last
	stq	r25,	-16(r16)	C L1 pair

	cmpult	u0,	v0,	cy1	C U1 did it borrow
	beq	r7,	$fix3		C U0 fix exact 0
$ret3:	ldq	u0,	32(r17)		C L0 get next ones
	ldq	v0,	32(r18)		C L1

	bis	r31,	r31,	r31	C L  damp out
	subq	r2,	r23,	r25	C U1 borrow from last
	bis	r31,	r31,	r31	C L  moves in L !
	subq	u1,	v1,	r5	C U0 sub two data

	beq	r2,	$fix4		C U1 fix exact zero
$ret4:	cmpult	u1,	v1,	cy0	C U0 did it borrow
	ldq	u1,	40(r17)		C L0 get next ones
	ldq	v1,	40(r18)		C L1

	subq	u0,	v0,	r8	C U1 sub two data
	stq	r24,	-8(r16)		C L0 store pair
	subq	r5,	cy1,	r24	C U0 borrow from last
	stq	r25,	0(r16)		C L1

	cmpult	u0,	v0,	r22	C U1 did it borrow
	beq	r5,	$fix5		C U0 fix exact zero
$ret5:	ldq	u0,	48(r17)		C L0 get next ones
	ldq	v0,	48(r18)		C L1

	ldl	r31, 256(r17)		C L0 prefetch
	subq	r8,	cy0,	r25	C U1 borrow from last
	ldl	r31, 256(r18)		C L1 prefetch
	subq	u1,	v1,	r7	C U0 sub two data

	beq	r8,	$fix6		C U1 fix exact zero
$ret6:	cmpult	u1,	v1,	r23	C U0 did it borrow
	ldq	u1,	56(r17)		C L0 get next ones
	ldq	v1,	56(r18)		C L1

	lda	r17,	64(r17)		C L0 move pointer
	bis	r31,	r31,	r31	C U
	lda	r18,	64(r18)		C L1 move pointer
	bge	r19,	$Loop		C U1 loop control
C ==== main loop end

$Lend:	subq	u0,	v0,	r2	C U1 sub two data
	stq	r24,	8(r16)		C L0 put an answer
	subq	r7,	r22,	r24	C U0 borrow from last
	stq	r25,	16(r16)		C L1 pair
	cmpult	u0,	v0,	cy1	C U1 did it borrow
	beq	r7,	$fix7c		C U0 fix exact 0
$ret7c:	subq	r2,	r23,	r25	C U1 borrow from last
	subq	u1,	v1,	r5	C U0 sub two data
	beq	r2,	$fix0c		C U1 fix exact zero
$ret0c:	cmpult	u1,	v1,	cy0	C U0 did it borrow
	stq	r24,	24(r16)		C L0 store pair
	subq	r5,	cy1,	r24	C U0 borrow from last
	stq	r25,	32(r16)		C L1
	beq	r5,	$fix1c		C U0 fix exact zero
$ret1c:	stq	r24,	40(r16)		C L0 put an answer
	lda	r16,	48(r16)		C L0 move pointer

	lda	r19,	8(r19)
	beq	r19,	$Lret

	ldq	u1,	0(r17)
	ldq	v1,	0(r18)
$Lsmall:
	lda	r19,	-1(r19)
	beq	r19,	$Lend0

	ALIGN(8)
$Loop0:	subq	u1,	v1,	r2	C main sub
	cmpult	u1,	v1,	r8	C compute bw from last sub
	ldq	u1,	8(r17)
	ldq	v1,	8(r18)
	subq	r2,	cy0,	r5	C borrow sub
	lda	r17,	8(r17)
	lda	r18,	8(r18)
	stq	r5,	0(r16)
	cmpult	r2,	cy0,	cy0	C compute bw from last sub
	lda	r19,	-1(r19)		C decr loop cnt
	bis	r8,	cy0,	cy0	C combine bw from the two subs
	lda	r16,	8(r16)
	bne	r19,	$Loop0
$Lend0:	subq	u1,	v1,	r2	C main sub
	subq	r2,	cy0,	r5	C borrow sub
	cmpult	u1,	v1,	r8	C compute bw from last sub
	cmpult	r2,	cy0,	cy0	C compute bw from last sub
	stq	r5,	0(r16)
	bis	r8,	cy0,	r0	C combine bw from the two subs
	ret	r31,(r26),1

	ALIGN(8)
$Lret:	lda	r0,	0(cy0)		C copy borrow into return register
	ret	r31,(r26),1

$fix5f:	bis	r23,	cy0,	r23	C bring forward borrow
	br	r31,	$ret5f
$fix6f:	bis	r22,	r23,	r22	C bring forward borrow
	br	r31,	$ret6f
$fix0:	bis	cy1,	r23,	cy1	C bring forward borrow
	br	r31,	$ret0
$fix1:	bis	cy0,	cy1,	cy0	C bring forward borrow
	br	r31,	$ret1
$fix2:	bis	r22,	cy0,	r22	C bring forward borrow
	br	r31,	$ret2
$fix3:	bis	r23,	r22,	r23	C bring forward borrow
	br	r31,	$ret3
$fix4:	bis	cy1,	r23,	cy1	C bring forward borrow
	br	r31,	$ret4
$fix5:	bis	cy1,	cy0,	cy0	C bring forward borrow
	br	r31,	$ret5
$fix6:	bis	r22,	cy0,	r22	C bring forward borrow
	br	r31,	$ret6
$fix7:	bis	r23,	r22,	r23	C bring forward borrow
	br	r31,	$ret7
$fix0c:	bis	cy1,	r23,	cy1	C bring forward borrow
	br	r31,	$ret0c
$fix1c:	bis	cy0,	cy1,	cy0	C bring forward borrow
	br	r31,	$ret1c
$fix7c:	bis	r23,	r22,	r23	C bring forward borrow
	br	r31,	$ret7c

EPILOGUE()
ASM_END()
