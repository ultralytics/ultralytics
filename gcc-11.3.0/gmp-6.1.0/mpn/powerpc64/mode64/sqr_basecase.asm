dnl  PowerPC-64 mpn_sqr_basecase.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 1999-2001, 2003-2006, 2008, 2010, 2011 Free Software Foundation,
dnl  Inc.

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
C POWER3/PPC630         6-18
C POWER4/PPC970          8
C POWER5                 8
C POWER6                16.25
C POWER7                 3.77

C NOTES
C  * This is very crude, cleanup!
C  * Try to reduce the number of needed live registers.
C  * Rewrite for POWER6 to use 8 consecutive muls, not 2 groups of 4.  The
C    cost will be more live registers.
C  * Rewrite for POWER7 to use addmul_2 building blocks; this will reduce code
C    size a lot and speed things up perhaps 25%.
C  * Use computed goto in order to compress the code.
C  * Implement a larger final corner.
C  * Schedule callee-saves register saves into other insns.  This could save
C    about 5 cycles/call.  (We cannot analogously optimise the restores, since
C    the sqr_diag_addlsh1 loop has no wind-down code as currently written.)
C  * Should the alternating std/adde sequences be split?  Some pipelines handle
C    adde poorly, and might sequentialise all these instructions.
C  * The sqr_diag_addlsh1 loop was written for POWER6 and its preferences for
C    adjacent integer multiply insns.  Except for the multiply insns, the code
C    was not carefully optimised for POWER6 or any other CPU.
C  * Perform cross-jumping in sqr_diag_addlsh1's feed-in code, into the loop.

C INPUT PARAMETERS
define(`rp', `r3')
define(`up', `r4')
define(`n',  `r5')

define(`rp_outer', `r25')
define(`up_outer', `r21')
define(`rp_saved', `r22')
define(`up_saved', `r23')
define(`n_saved',  `r24')

ASM_START()
PROLOGUE(mpn_sqr_basecase)
	cmpdi	cr0, n, 2
	bge	cr0, L(ge2)
	ld	r5, 0(up)	C n = 1
	nop
	mulld	r8, r5, r5	C weight 0
	mulhdu	r9, r5, r5	C weight 1
	std	r8, 0(rp)
	std	r9, 8(rp)
	blr
	ALIGN(16)
L(ge2):	bgt	cr0, L(gt2)
	ld	r0, 0(up)	C n = 2
	nop
	mulld	r8, r0, r0	C u0 * u0
	mulhdu	r9, r0, r0	C u0 * u0
	ld	r6, 8(up)
	mulld	r10, r6, r6	C u1 * u1
	mulhdu	r11, r6, r6	C u1 * u1
	mulld	r4, r6, r0	C u1 * u0
	mulhdu	r5, r6, r0	C u1 * u0
	addc	r4, r4, r4
	adde	r5, r5, r5
	addze	r11, r11
	addc	r9, r9, r4
	adde	r10, r10, r5
	addze	r11, r11
	std	r8, 0(rp)
	std	r9, 8(rp)
	std	r10, 16(rp)
	std	r11, 24(rp)
	blr

	ALIGN(16)
L(gt2):	std	r31,  -8(r1)
	std	r30, -16(r1)
	std	r29, -24(r1)
	std	r28, -32(r1)
	std	r27, -40(r1)
	std	r26, -48(r1)
	std	r25, -56(r1)
	std	r24, -64(r1)
	std	r23, -72(r1)
	std	r22, -80(r1)
	std	r21, -88(r1)

	mr	rp_saved, rp
	mr	up_saved, up
	mr	n_saved, n
	mr	rp_outer, rp
	mr	up_outer, up

	rldicl.	r0, n, 0,62	C r0 = n & 3, set cr0
	cmpdi	cr6, r0, 2
	addic	r7, n, 2	C compute count...
	srdi	r7, r7, 2	C ...for ctr
	mtctr	r7		C copy count into ctr
	beq-	cr0, L(b0)
	blt-	cr6, L(b1)
	beq-	cr6, L(b2)

L(b3):	ld	r6, 0(up)
	ld	r9, 8(up)
	ld	r27, 16(up)
	addi	up, up, 24
	li	r12, 0		C carry limb
	bdz	L(em3)

	ALIGN(16)
L(tm3):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r9, 0(up)
	ld	r27, 8(up)
	adde	r0, r0, r12
	adde	r7, r7, r26
	mulld	r26, r9, r6
	mulhdu	r10, r9, r6
	mulld	r11, r27, r6
	mulhdu	r12, r27, r6
	ld	r9, 16(up)
	ld	r27, 24(up)
	std	r0, 8(rp)
	adde	r26, r26, r8
	std	r7, 16(rp)
	adde	r11, r11, r10
	std	r26, 24(rp)
	addi	up, up, 32
	std	r11, 32(rp)
	addi	rp, rp, 32
	bdnz	L(tm3)

L(em3):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	adde	r0, r0, r12
	adde	r7, r7, r26
	std	r0, 8(rp)
	std	r7, 16(rp)
	addze	r8, r8
	std	r8, 24(rp)
	addi	n, n, 2
	b	L(outer_loop)

L(b0):	ld	r6, 0(up)
	ld	r27, 8(up)
	mulld	r7, r27, r6
	mulhdu	r12, r27, r6
	std	r7, 8(rp)
	addi	rp, rp, 8
	ld	r9, 16(up)
	ld	r27, 24(up)
	addi	up, up, 32
	bdz	L(em0)

	ALIGN(16)
L(tm0):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r9, 0(up)
	ld	r27, 8(up)
	adde	r0, r0, r12
	adde	r7, r7, r26
	mulld	r26, r9, r6
	mulhdu	r10, r9, r6
	mulld	r11, r27, r6
	mulhdu	r12, r27, r6
	ld	r9, 16(up)
	ld	r27, 24(up)
	std	r0, 8(rp)
	adde	r26, r26, r8
	std	r7, 16(rp)
	adde	r11, r11, r10
	std	r26, 24(rp)
	addi	up, up, 32
	std	r11, 32(rp)
	addi	rp, rp, 32
	bdnz	L(tm0)

L(em0):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	adde	r0, r0, r12
	adde	r7, r7, r26
	std	r0, 8(rp)
	std	r7, 16(rp)
	addze	r8, r8
	std	r8, 24(rp)
	addi	n, n, 2
	b	L(outer_loop_ent_2)

L(b1):	ld	r6, 0(up)
	ld	r9, 8(up)
	ld	r27, 16(up)
	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r12, r27, r6
	addc	r7, r7, r26
	std	r0, 8(rp)
	std	r7, 16(rp)
	addi	rp, rp, 16
	ld	r9, 24(up)
	ld	r27, 32(up)
	addi	up, up, 40
	bdz	L(em1)

	ALIGN(16)
L(tm1):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r9, 0(up)
	ld	r27, 8(up)
	adde	r0, r0, r12
	adde	r7, r7, r26
	mulld	r26, r9, r6
	mulhdu	r10, r9, r6
	mulld	r11, r27, r6
	mulhdu	r12, r27, r6
	ld	r9, 16(up)
	ld	r27, 24(up)
	std	r0, 8(rp)
	adde	r26, r26, r8
	std	r7, 16(rp)
	adde	r11, r11, r10
	std	r26, 24(rp)
	addi	up, up, 32
	std	r11, 32(rp)
	addi	rp, rp, 32
	bdnz	L(tm1)

L(em1):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	adde	r0, r0, r12
	adde	r7, r7, r26
	std	r0, 8(rp)
	std	r7, 16(rp)
	addze	r8, r8
	std	r8, 24(rp)
	addi	n, n, 2
	b	L(outer_loop_ent_3)

L(b2):	addi	r7, r7, -1	C FIXME
	mtctr	r7		C FIXME
	ld	r6, 0(up)
	ld	r9, 8(up)
	ld	r27, 16(up)
	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r9, 24(up)
	mulld	r11, r9, r6
	mulhdu	r10, r9, r6
	addc	r7, r7, r26
	adde	r11, r11, r8
	addze	r12, r10
	std	r0, 8(rp)
	std	r7, 16(rp)
	std	r11, 24(rp)
	addi	rp, rp, 24
	ld	r9, 32(up)
	ld	r27, 40(up)
	addi	up, up, 48
	bdz	L(em2)

	ALIGN(16)
L(tm2):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r9, 0(up)
	ld	r27, 8(up)
	adde	r0, r0, r12
	adde	r7, r7, r26
	mulld	r26, r9, r6
	mulhdu	r10, r9, r6
	mulld	r11, r27, r6
	mulhdu	r12, r27, r6
	ld	r9, 16(up)
	ld	r27, 24(up)
	std	r0, 8(rp)
	adde	r26, r26, r8
	std	r7, 16(rp)
	adde	r11, r11, r10
	std	r26, 24(rp)
	addi	up, up, 32
	std	r11, 32(rp)
	addi	rp, rp, 32
	bdnz	L(tm2)

L(em2):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	adde	r0, r0, r12
	adde	r7, r7, r26
	std	r0, 8(rp)
	std	r7, 16(rp)
	addze	r8, r8
	std	r8, 24(rp)
	addi	n, n, 2
	b	L(outer_loop_ent_0)


L(outer_loop):
	addi	n, n, -1
	addi	up_outer, up_outer, 8
	addi	rp_outer, rp_outer, 16

	mr	up, up_outer
	addi	rp, rp_outer, 8

	srdi	r0, n, 2
	mtctr	r0

	bdz	L(outer_end)

	ld	r6, 0(up)
	ld	r9, 8(up)
	ld	r27, 16(up)
	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r9, 24(up)
	ld	r28, 0(rp)
	ld	r29, 8(rp)
	ld	r30, 16(rp)
	mulld	r11, r9, r6
	mulhdu	r10, r9, r6
	addc	r7, r7, r26
	adde	r11, r11, r8
	addze	r12, r10
	addc	r0, r0, r28
	std	r0, 0(rp)
	adde	r7, r7, r29
	std	r7, 8(rp)
	adde	r11, r11, r30
	std	r11, 16(rp)
	addi	rp, rp, 24
	ld	r9, 32(up)
	ld	r27, 40(up)
	addi	up, up, 48
	bdz	L(ea1)

	ALIGN(16)
L(ta1):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6	C 9
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6	C 27
	ld	r9, 0(up)
	ld	r28, 0(rp)
	ld	r27, 8(up)
	ld	r29, 8(rp)
	adde	r0, r0, r12	C 0 12
	adde	r7, r7, r26	C 5 7
	mulld	r26, r9, r6
	mulhdu	r10, r9, r6	C 9
	mulld	r11, r27, r6
	mulhdu	r12, r27, r6	C 27
	ld	r9, 16(up)
	ld	r30, 16(rp)
	ld	r27, 24(up)
	ld	r31, 24(rp)
	adde	r26, r26, r8	C 8 5
	adde	r11, r11, r10	C 10 11
	addze	r12, r12	C 12
	addc	r0, r0, r28	C 0 28
	std	r0, 0(rp)	C 0
	adde	r7, r7, r29	C 7 29
	std	r7, 8(rp)	C 7
	adde	r26, r26, r30	C 5 30
	std	r26, 16(rp)	C 5
	adde	r11, r11, r31	C 11 31
	std	r11, 24(rp)	C 11
	addi	up, up, 32
	addi	rp, rp, 32
	bdnz	L(ta1)

L(ea1):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r28, 0(rp)
	ld	r29, 8(rp)
	adde	r0, r0, r12
	adde	r7, r7, r26
	addze	r8, r8
	addc	r0, r0, r28
	std	r0, 0(rp)
	adde	r7, r7, r29
	std	r7, 8(rp)
	addze	r8, r8
	std	r8, 16(rp)

L(outer_loop_ent_0):
	addi	n, n, -1
	addi	up_outer, up_outer, 8
	addi	rp_outer, rp_outer, 16

	mr	up, up_outer
	addi	rp, rp_outer, 8

	srdi	r0, n, 2
	mtctr	r0

	ld	r6, 0(up)
	ld	r9, 8(up)
	ld	r27, 16(up)
	ld	r28, 0(rp)
	ld	r29, 8(rp)
	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	addc	r0, r0, r28
	adde	r7, r7, r26
	addze	r12, r8
	std	r0, 0(rp)
	adde	r7, r7, r29
	std	r7, 8(rp)
	addi	rp, rp, 16
	ld	r9, 24(up)
	ld	r27, 32(up)
	addi	up, up, 40
	bdz	L(ea0)

	ALIGN(16)
L(ta0):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6	C 9
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6	C 27
	ld	r9, 0(up)
	ld	r28, 0(rp)
	ld	r27, 8(up)
	ld	r29, 8(rp)
	adde	r0, r0, r12	C 0 12
	adde	r7, r7, r26	C 5 7
	mulld	r26, r9, r6
	mulhdu	r10, r9, r6	C 9
	mulld	r11, r27, r6
	mulhdu	r12, r27, r6	C 27
	ld	r9, 16(up)
	ld	r30, 16(rp)
	ld	r27, 24(up)
	ld	r31, 24(rp)
	adde	r26, r26, r8	C 8 5
	adde	r11, r11, r10	C 10 11
	addze	r12, r12	C 12
	addc	r0, r0, r28	C 0 28
	std	r0, 0(rp)	C 0
	adde	r7, r7, r29	C 7 29
	std	r7, 8(rp)	C 7
	adde	r26, r26, r30	C 5 30
	std	r26, 16(rp)	C 5
	adde	r11, r11, r31	C 11 31
	std	r11, 24(rp)	C 11
	addi	up, up, 32
	addi	rp, rp, 32
	bdnz	L(ta0)

L(ea0):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r28, 0(rp)
	ld	r29, 8(rp)
	adde	r0, r0, r12
	adde	r7, r7, r26
	addze	r8, r8
	addc	r0, r0, r28
	std	r0, 0(rp)
	adde	r7, r7, r29
	std	r7, 8(rp)
	addze	r8, r8
	std	r8, 16(rp)

L(outer_loop_ent_3):
	addi	n, n, -1
	addi	up_outer, up_outer, 8
	addi	rp_outer, rp_outer, 16

	mr	up, up_outer
	addi	rp, rp_outer, 8

	srdi	r0, n, 2
	mtctr	r0

	ld	r6, 0(up)
	ld	r9, 8(up)
	ld	r28, 0(rp)
	mulld	r0, r9, r6
	mulhdu	r12, r9, r6
	addc	r0, r0, r28
	std	r0, 0(rp)
	addi	rp, rp, 8
	ld	r9, 16(up)
	ld	r27, 24(up)
	addi	up, up, 32
	bdz	L(ea3)

	ALIGN(16)
L(ta3):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6	C 9
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6	C 27
	ld	r9, 0(up)
	ld	r28, 0(rp)
	ld	r27, 8(up)
	ld	r29, 8(rp)
	adde	r0, r0, r12	C 0 12
	adde	r7, r7, r26	C 5 7
	mulld	r26, r9, r6
	mulhdu	r10, r9, r6	C 9
	mulld	r11, r27, r6
	mulhdu	r12, r27, r6	C 27
	ld	r9, 16(up)
	ld	r30, 16(rp)
	ld	r27, 24(up)
	ld	r31, 24(rp)
	adde	r26, r26, r8	C 8 5
	adde	r11, r11, r10	C 10 11
	addze	r12, r12	C 12
	addc	r0, r0, r28	C 0 28
	std	r0, 0(rp)	C 0
	adde	r7, r7, r29	C 7 29
	std	r7, 8(rp)	C 7
	adde	r26, r26, r30	C 5 30
	std	r26, 16(rp)	C 5
	adde	r11, r11, r31	C 11 31
	std	r11, 24(rp)	C 11
	addi	up, up, 32
	addi	rp, rp, 32
	bdnz	L(ta3)

L(ea3):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r28, 0(rp)
	ld	r29, 8(rp)
	adde	r0, r0, r12
	adde	r7, r7, r26
	addze	r8, r8
	addc	r0, r0, r28
	std	r0, 0(rp)
	adde	r7, r7, r29
	std	r7, 8(rp)
	addze	r8, r8
	std	r8, 16(rp)


L(outer_loop_ent_2):
	addi	n, n, -1
	addi	up_outer, up_outer, 8
	addi	rp_outer, rp_outer, 16

	mr	up, up_outer
	addi	rp, rp_outer, 8

	srdi	r0, n, 2
	mtctr	r0

	addic	r0, r0, 0
	li	r12, 0		C cy_limb = 0
	ld	r6, 0(up)
	ld	r9, 8(up)
	ld	r27, 16(up)
	bdz	L(ea2)
	addi	up, up, 24

	ALIGN(16)
L(ta2):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6	C 9
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6	C 27
	ld	r9, 0(up)
	ld	r28, 0(rp)
	ld	r27, 8(up)
	ld	r29, 8(rp)
	adde	r0, r0, r12	C 0 12
	adde	r7, r7, r26	C 5 7
	mulld	r26, r9, r6
	mulhdu	r10, r9, r6	C 9
	mulld	r11, r27, r6
	mulhdu	r12, r27, r6	C 27
	ld	r9, 16(up)
	ld	r30, 16(rp)
	ld	r27, 24(up)
	ld	r31, 24(rp)
	adde	r26, r26, r8	C 8 5
	adde	r11, r11, r10	C 10 11
	addze	r12, r12	C 12
	addc	r0, r0, r28	C 0 28
	std	r0, 0(rp)	C 0
	adde	r7, r7, r29	C 7 29
	std	r7, 8(rp)	C 7
	adde	r26, r26, r30	C 5 30
	std	r26, 16(rp)	C 5
	adde	r11, r11, r31	C 11 31
	std	r11, 24(rp)	C 11
	addi	up, up, 32
	addi	rp, rp, 32
	bdnz	L(ta2)

L(ea2):	mulld	r0, r9, r6
	mulhdu	r26, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r28, 0(rp)
	ld	r29, 8(rp)
	adde	r0, r0, r12
	adde	r7, r7, r26
	addze	r8, r8
	addc	r0, r0, r28
	std	r0, 0(rp)
	adde	r7, r7, r29
	std	r7, 8(rp)
	addze	r8, r8
	std	r8, 16(rp)

	b	L(outer_loop)

L(outer_end):
	ld	r6, 0(up)
	ld	r9, 8(up)
	ld	r11, 0(rp)
	mulld	r0, r9, r6
	mulhdu	r8, r9, r6
	addc	r0, r0, r11
	std	r0, 0(rp)
	addze	r8, r8
	std	r8, 8(rp)

define(`rp',  `rp_saved')
define(`up',  `r5')
define(`n',   `r6')
define(`climb',	`r0')

	addi	r4, rp_saved, 8
	mr	r5, up_saved
	mr	r6, n_saved

	rldicl.	r0, n, 0,62		C r0 = n & 3, set cr0
	cmpdi	cr6, r0, 2
	addi	n, n, 2			C compute count...
	srdi	n, n, 2			C ...for ctr
	mtctr	n			C put loop count into ctr
	beq	cr0, L(xb0)
	blt	cr6, L(xb1)
	beq	cr6, L(xb2)

L(xb3):	ld	r6,   0(up)
	ld	r7,   8(up)
	ld	r12, 16(up)
	addi	up, up, 24
	mulld	r24, r6, r6
	mulhdu	r25, r6, r6
	mulld	r26, r7, r7
	mulhdu	r27, r7, r7
	mulld	r28, r12, r12
	mulhdu	r29, r12, r12
	ld	r10,  8(rp)
	ld	r11, 16(rp)
	ld	r6,  24(rp)
	ld	r7,  32(rp)
	addc	r10, r10, r10
	adde	r11, r11, r11
	adde	r6, r6, r6
	adde	r7, r7, r7
	addze	climb, r29
	addc	r10, r10, r25
	adde	r11, r11, r26
	adde	r6, r6, r27
	adde	r7, r7, r28
	std	r24,  0(rp)
	std	r10,  8(rp)
	std	r11, 16(rp)
	std	r6,  24(rp)
	std	r7,  32(rp)
	addi	rp, rp, 40
	bdnz	L(top)
	b	L(end)

L(xb2):	ld	r6,  0(up)
	ld	r7,  8(up)
	addi	up, up, 16
	mulld	r24, r6, r6
	mulhdu	r25, r6, r6
	mulld	r26, r7, r7
	mulhdu	r27, r7, r7
	ld	r10,  8(rp)
	ld	r11, 16(rp)
	addc	r10, r10, r10
	adde	r11, r11, r11
	addze	climb, r27
	addc	r10, r10, r25
	adde	r11, r11, r26
	std	r24,  0(rp)
	std	r10,  8(rp)
	std	r11, 16(rp)
	addi	rp, rp, 24
	bdnz	L(top)
	b	L(end)

L(xb0):	ld	r6,   0(up)
	ld	r7,   8(up)
	ld	r12, 16(up)
	ld	r23, 24(up)
	addi	up, up, 32
	mulld	r24, r6, r6
	mulhdu	r25, r6, r6
	mulld	r26, r7, r7
	mulhdu	r27, r7, r7
	mulld	r28, r12, r12
	mulhdu	r29, r12, r12
	mulld	r30, r23, r23
	mulhdu	r31, r23, r23
	ld	r10,  8(rp)
	ld	r11, 16(rp)
	ld	r6,  24(rp)
	ld	r7,  32(rp)
	ld	r12, 40(rp)
	ld	r23, 48(rp)
	addc	r10, r10, r10
	adde	r11, r11, r11
	adde	r6, r6, r6
	adde	r7, r7, r7
	adde	r12, r12, r12
	adde	r23, r23, r23
	addze	climb, r31
	std	r24,  0(rp)
	addc	r10, r10, r25
	std	r10,  8(rp)
	adde	r11, r11, r26
	std	r11, 16(rp)
	adde	r6, r6, r27
	std	r6,  24(rp)
	adde	r7, r7, r28
	std	r7,  32(rp)
	adde	r12, r12, r29
	std	r12, 40(rp)
	adde	r23, r23, r30
	std	r23, 48(rp)
	addi	rp, rp, 56
	bdnz	L(top)
	b	L(end)

L(xb1):	ld	r6,  0(up)
	addi	up, up, 8
	mulld	r24, r6, r6
	mulhdu	climb, r6, r6
	std	r24, 0(rp)
	addic	rp, rp, 8		C clear carry as side-effect

	ALIGN(32)
L(top):	ld	r6,   0(up)
	ld	r7,   8(up)
	ld	r12, 16(up)
	ld	r23, 24(up)
	addi	up, up, 32
	mulld	r24, r6, r6
	mulhdu	r25, r6, r6
	mulld	r26, r7, r7
	mulhdu	r27, r7, r7
	mulld	r28, r12, r12
	mulhdu	r29, r12, r12
	mulld	r30, r23, r23
	mulhdu	r31, r23, r23
	ld	r8,   0(rp)
	ld	r9,   8(rp)
	adde	r8, r8, r8
	adde	r9, r9, r9
	ld	r10, 16(rp)
	ld	r11, 24(rp)
	adde	r10, r10, r10
	adde	r11, r11, r11
	ld	r6,  32(rp)
	ld	r7,  40(rp)
	adde	r6, r6, r6
	adde	r7, r7, r7
	ld	r12, 48(rp)
	ld	r23, 56(rp)
	adde	r12, r12, r12
	adde	r23, r23, r23
	addze	r31, r31
	addc	r8, r8, climb
	std	r8,   0(rp)
	adde	r9, r9, r24
	std	r9,   8(rp)
	adde	r10, r10, r25
	std	r10, 16(rp)
	adde	r11, r11, r26
	std	r11, 24(rp)
	adde	r6, r6, r27
	std	r6,  32(rp)
	adde	r7, r7, r28
	std	r7,  40(rp)
	adde	r12, r12, r29
	std	r12, 48(rp)
	adde	r23, r23, r30
	std	r23, 56(rp)
	mr	climb, r31
	addi	rp, rp, 64
	bdnz	L(top)

L(end):	addze	climb, climb
	std	climb,  0(rp)

	ld	r31,  -8(r1)
	ld	r30, -16(r1)
	ld	r29, -24(r1)
	ld	r28, -32(r1)
	ld	r27, -40(r1)
	ld	r26, -48(r1)
	ld	r25, -56(r1)
	ld	r24, -64(r1)
	ld	r23, -72(r1)
	ld	r22, -80(r1)
	ld	r21, -88(r1)
	blr
EPILOGUE()
