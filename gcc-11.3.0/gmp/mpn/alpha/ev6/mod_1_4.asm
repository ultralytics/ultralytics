dnl Alpha mpn_mod_1s_4p

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2009, 2010 Free Software Foundation, Inc.

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

C TODO:
C  * Optimise.  2.75 c/l should be possible.
C  * Write a proper mpn_mod_1s_4p_cps.  The code below was compiler generated.
C  * Optimise feed-in code, starting the sw pipeline in switch code.
C  * Shorten software pipeline.  The mul instructions are scheduled too far
C    from their users.  Fixing this will allow us to use fewer registers.
C  * If we cannot reduce register usage, write perhaps small-n basecase.
C  * Does this work for PIC?

C      cycles/limb
C EV4:     ?
C EV5:    23
C EV6:     3

define(`ap',     `r16')
define(`n',      `r17')
define(`pl',     `r24')
define(`ph',     `r25')
define(`rl',     `r6')
define(`rh',     `r7')
define(`B1modb', `r1')
define(`B2modb', `r2')
define(`B3modb', `r3')
define(`B4modb', `r4')
define(`B5modb', `r5')

ASM_START()
PROLOGUE(mpn_mod_1s_4p)
	lda	r30, -64(r30)
	stq	r9, 8(r30)
	ldq	B1modb, 16(r19)
	stq	r10, 16(r30)
	ldq	B2modb, 24(r19)
	stq	r11, 24(r30)
	ldq	B3modb, 32(r19)
	stq	r12, 32(r30)
	ldq	B4modb, 40(r19)
	stq	r13, 40(r30)
	ldq	B5modb, 48(r19)
	s8addq	n, ap, ap		C point ap at vector end

	and	n, 3, r0
	lda	n, -4(n)
	beq	r0, L(b0)
	lda	r6, -2(r0)
	blt	r6, L(b1)
	beq	r6, L(b2)

L(b3):	ldq	r21, -16(ap)
	ldq	r22, -8(ap)
	ldq	r20, -24(ap)
	mulq	r21, B1modb, r8
	umulh	r21, B1modb, r12
	mulq	r22, B2modb, r9
	umulh	r22, B2modb, r13
	addq	r8, r20, pl
	cmpult	pl, r8, r0
	addq	r0, r12, ph
	addq	r9, pl, rl
	cmpult	rl, r9, r0
	addq	r13, ph, ph
	addq	r0, ph, rh
	lda	ap, -56(ap)
	br	L(com)

L(b0):	ldq	r21, -24(ap)
	ldq	r22, -16(ap)
	ldq	r23, -8(ap)
	ldq	r20, -32(ap)
	mulq	r21, B1modb, r8
	umulh	r21, B1modb, r12
	mulq	r22, B2modb, r9
	umulh	r22, B2modb, r13
	mulq	r23, B3modb, r10
	umulh	r23, B3modb, r27
	addq	r8, r20, pl
	cmpult	pl, r8, r0
	addq	r0, r12, ph
	addq	r9, pl, pl
	cmpult	pl, r9, r0
	addq	r13, ph, ph
	addq	r0, ph, ph
	addq	r10, pl, rl
	cmpult	rl, r10, r0
	addq	r27, ph, ph
	addq	r0, ph, rh
	lda	ap, -64(ap)
	br	L(com)

L(b1):	bis	r31, r31, rh
	ldq	rl, -8(ap)
	lda	ap, -40(ap)
	br	L(com)

L(b2):	ldq	rh, -8(ap)
	ldq	rl, -16(ap)
	lda	ap, -48(ap)

L(com):	ble	n, L(ed3)
	ldq	r21, 8(ap)
	ldq	r22, 16(ap)
	ldq	r23, 24(ap)
	ldq	r20, 0(ap)
	lda	n, -4(n)
	lda	ap, -32(ap)
	mulq	r21, B1modb, r8
	umulh	r21, B1modb, r12
	mulq	r22, B2modb, r9
	umulh	r22, B2modb, r13
	mulq	r23, B3modb, r10
	umulh	r23, B3modb, r27
	mulq	rl, B4modb, r11
	umulh	rl, B4modb, r28
	ble	n, L(ed2)

	ALIGN(16)
L(top):	ldq	r21, 8(ap)
	mulq	rh, B5modb, rl
	addq	r8, r20, pl
	ldq	r22, 16(ap)
	cmpult	pl, r8, r0
	umulh	rh, B5modb, rh
	ldq	r23, 24(ap)
	addq	r0, r12, ph
	addq	r9, pl, pl
	mulq	r21, B1modb, r8
	cmpult	pl, r9, r0
	addq	r13, ph, ph
	umulh	r21, B1modb, r12
	lda	ap, -32(ap)
	addq	r0, ph, ph
	addq	r10, pl, pl
	mulq	r22, B2modb, r9
	cmpult	pl, r10, r0
	addq	r27, ph, ph
	addq	r11, pl, pl
	umulh	r22, B2modb, r13
	addq	r0, ph, ph
	cmpult	pl, r11, r0
	addq	r28, ph, ph
	mulq	r23, B3modb, r10
	ldq	r20, 32(ap)
	addq	pl, rl, rl
	umulh	r23, B3modb, r27
	addq	r0, ph, ph
	cmpult	rl, pl, r0
	mulq	rl, B4modb, r11
	addq	ph, rh, rh
	umulh	rl, B4modb, r28
	addq	r0, rh, rh
	lda	n, -4(n)
	bgt	n, L(top)

L(ed2):	mulq	rh, B5modb, rl
	addq	r8, r20, pl
	umulh	rh, B5modb, rh
	cmpult	pl, r8, r0
	addq	r0, r12, ph
	addq	r9, pl, pl
	cmpult	pl, r9, r0
	addq	r13, ph, ph
	addq	r0, ph, ph
	addq	r10, pl, pl
	cmpult	pl, r10, r0
	addq	r27, ph, ph
	addq	r11, pl, pl
	addq	r0, ph, ph
	cmpult	pl, r11, r0
	addq	r28, ph, ph
	addq	pl, rl, rl
	addq	r0, ph, ph
	cmpult	rl, pl, r0
	addq	ph, rh, rh
	addq	r0, rh, rh

L(ed3):	mulq	rh, B1modb, r8
	umulh	rh, B1modb, rh
	addq	r8, rl, rl
	cmpult	rl, r8, r0
	addq	r0, rh, rh

	ldq	r24, 8(r19)		C cnt
	sll	rh, r24, rh
	subq	r31, r24, r25
	srl	rl, r25, r2
	sll	rl, r24, rl
	or	r2, rh, rh

	ldq	r23, 0(r19)		C bi
	mulq	rh, r23, r8
	umulh	rh, r23, r9
	addq	rh, 1, r7
	addq	r8, rl, r8		C ql
	cmpult	r8, rl, r0
	addq	r9, r7, r9
	addq	r0, r9, r9		C qh
	mulq	r9, r18, r21		C qh * b
	subq	rl, r21, rl
	cmpult	r8, rl, r0		C rl > ql
	negq	r0, r0
	and	r0, r18, r0
	addq	rl, r0, rl
	cmpule	r18, rl, r0		C rl >= b
	negq	r0, r0
	and	r0, r18, r0
	subq	rl, r0, rl

	srl	rl, r24, r0

	ldq	r9, 8(r30)
	ldq	r10, 16(r30)
	ldq	r11, 24(r30)
	ldq	r12, 32(r30)
	ldq	r13, 40(r30)
	lda	r30, 64(r30)
	ret	r31, (r26), 1
EPILOGUE()

PROLOGUE(mpn_mod_1s_4p_cps,gp)
	lda	r30, -32(r30)
	stq	r26, 0(r30)
	stq	r9, 8(r30)
	stq	r10, 16(r30)
	stq	r11, 24(r30)
	mov	r16, r11
	LEA(	r4, __clz_tab)
	lda	r10, 65(r31)
	cmpbge	r31, r17, r1
	srl	r1, 1, r1
	xor	r1, 127, r1
	addq	r1, r4, r1
	ldq_u	r2, 0(r1)
	extbl	r2, r1, r2
	s8subq	r2, 7, r2
	srl	r17, r2, r3
	subq	r10, r2, r10
	addq	r3, r4, r3
	ldq_u	r1, 0(r3)
	extbl	r1, r3, r1
	subq	r10, r1, r10
	sll	r17, r10, r9
	mov	r9, r16
	jsr	r26, mpn_invert_limb
	LDGP(	r29, 0(r26))
	subq	r31, r10, r2
	lda	r1, 1(r31)
	sll	r1, r10, r1
	subq	r31, r9, r3
	srl	r0, r2, r2
	ldq	r26, 0(r30)
	bis	r2, r1, r2
	stq	r0, 0(r11)
	stq	r10, 8(r11)
	mulq	r2, r3, r2
	srl	r2, r10, r3
	umulh	r2, r0, r1
	stq	r3, 16(r11)
	mulq	r2, r0, r3
	ornot	r31, r1, r1
	subq	r1, r2, r1
	mulq	r1, r9, r1
	addq	r1, r9, r2
	cmpule	r1, r3, r3
	cmoveq	r3, r2, r1
	srl	r1, r10, r3
	umulh	r1, r0, r2
	stq	r3, 24(r11)
	mulq	r1, r0, r3
	ornot	r31, r2, r2
	subq	r2, r1, r2
	mulq	r2, r9, r2
	addq	r2, r9, r1
	cmpule	r2, r3, r3
	cmoveq	r3, r1, r2
	srl	r2, r10, r1
	umulh	r2, r0, r3
	stq	r1, 32(r11)
	mulq	r2, r0, r1
	ornot	r31, r3, r3
	subq	r3, r2, r3
	mulq	r3, r9, r3
	addq	r3, r9, r2
	cmpule	r3, r1, r1
	cmoveq	r1, r2, r3
	srl	r3, r10, r2
	umulh	r3, r0, r1
	stq	r2, 40(r11)
	mulq	r3, r0, r0
	ornot	r31, r1, r1
	subq	r1, r3, r1
	mulq	r1, r9, r1
	addq	r1, r9, r9
	cmpule	r1, r0, r0
	cmoveq	r0, r9, r1
	ldq	r9, 8(r30)
	srl	r1, r10, r1
	ldq	r10, 16(r30)
	stq	r1, 48(r11)
	ldq	r11, 24(r30)
	lda	r30, 32(r30)
	ret	r31, (r26), 1
EPILOGUE()
