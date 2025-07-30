dnl  PowerPC-64 mpn_mul_basecase.

dnl  Copyright 1999-2001, 2003-2006, 2008 Free Software Foundation, Inc.

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
C POWER6                24

C INPUT PARAMETERS
define(`rp', `r3')
define(`up', `r4')
define(`un', `r5')
define(`vp', `r6')
define(`vn', `r7')

define(`v0',	   `r25')
define(`outer_rp', `r22')
define(`outer_up', `r23')

ASM_START()
PROLOGUE(mpn_mul_basecase)

C Special code for un <= 2, for efficiency of these important cases,
C and since it simplifies the default code.
	cmpdi	cr0, un, 2
	bgt	cr0, L(un_gt2)
	cmpdi	cr6, vn, 1
	ld	r7, 0(vp)
	ld	r5, 0(up)
	mulld	r8, r5, r7	C weight 0
	mulhdu	r9, r5, r7	C weight 1
	std	r8, 0(rp)
	beq	cr0, L(2x)
	std	r9, 8(rp)
	blr
	ALIGN(16)
L(2x):	ld	r0, 8(up)
	mulld	r8, r0, r7	C weight 1
	mulhdu	r10, r0, r7	C weight 2
	addc	r9, r9, r8
	addze	r10, r10
	bne	cr6, L(2x2)
	std	r9, 8(rp)
	std	r10, 16(rp)
	blr
	ALIGN(16)
L(2x2):	ld	r6, 8(vp)
	nop
	mulld	r8, r5, r6	C weight 1
	mulhdu	r11, r5, r6	C weight 2
	addc	r9, r9, r8
	std	r9, 8(rp)
	adde	r11, r11, r10
	mulld	r12, r0, r6	C weight 2
	mulhdu	r0, r0, r6	C weight 3
	addze	r0, r0
	addc	r11, r11, r12
	addze	r0, r0
	std	r11, 16(rp)
	std	r0, 24(rp)
	blr

L(un_gt2):
	std	r31, -8(r1)
	std	r30, -16(r1)
	std	r29, -24(r1)
	std	r28, -32(r1)
	std	r27, -40(r1)
	std	r26, -48(r1)
	std	r25, -56(r1)
	std	r24, -64(r1)
	std	r23, -72(r1)
	std	r22, -80(r1)

	mr	outer_rp, rp
	mr	outer_up, up

	ld	v0, 0(vp)	C new v limb
	addi	vp, vp, 8
	ld	r26, 0(up)

	rldicl.	r0, un, 0,62	C r0 = n & 3, set cr0
	cmpdi	cr6, r0, 2
	addi	un, un, 1	C compute count...
	srdi	un, un, 2	C ...for ctr
	mtctr	un		C copy inner loop count into ctr
	beq	cr0, L(b0)
	blt	cr6, L(b1)
	beq	cr6, L(b2)


	ALIGN(16)
L(b3):	mulld	r0, r26, v0
	mulhdu	r12, r26, v0
	addic	r0, r0, 0
	std	r0, 0(rp)
	ld	r26, 8(up)
	ld	r27, 16(up)
	bdz	L(end_m_3)

	ALIGN(16)
L(lo_m_3):
	mulld	r0, r26, v0
	mulhdu	r31, r26, v0
	ld	r26, 24(up)
	nop
	mulld	r24, r27, v0
	mulhdu	r8, r27, v0
	ld	r27, 32(up)
	nop
	adde	r0, r0, r12
	adde	r24, r24, r31
	mulld	r9, r26, v0
	mulhdu	r10, r26, v0
	ld	r26, 40(up)
	nop
	mulld	r11, r27, v0
	mulhdu	r12, r27, v0
	ld	r27, 48(up)
	std	r0, 8(rp)
	adde	r9, r9, r8
	std	r24, 16(rp)
	adde	r11, r11, r10
	std	r9, 24(rp)
	addi	up, up, 32
	std	r11, 32(rp)
	addi	rp, rp, 32
	bdnz	L(lo_m_3)

	ALIGN(16)
L(end_m_3):
	mulld	r0, r26, v0
	mulhdu	r31, r26, v0

	mulld	r24, r27, v0
	mulhdu	r8, r27, v0

	adde	r0, r0, r12
	adde	r24, r24, r31

	std	r0, 8(rp)
	std	r24, 16(rp)
	addze	r8, r8
	std	r8, 24(rp)
	addic.	vn, vn, -1
	beq	L(ret)

	ALIGN(16)
L(outer_lo_3):
	mtctr	un		C copy inner loop count into ctr
	addi	rp, outer_rp, 8
	mr	up, outer_up
	addi	outer_rp, outer_rp, 8
	ld	v0, 0(vp)	C new v limb
	addi	vp, vp, 8
	ld	r26, 0(up)
	ld	r28, 0(rp)
	mulld	r0, r26, v0
	mulhdu	r12, r26, v0
	addc	r0, r0, r28
	std	r0, 0(rp)
	ld	r26, 8(up)
	ld	r27, 16(up)
	bdz	L(end_3)

	ALIGN(16)		C registers dying
L(lo_3):
	mulld	r0, r26, v0	C
	mulhdu	r10, r26, v0	C 26
	ld	r26, 24(up)	C
	ld	r28, 8(rp)	C
	mulld	r24, r27, v0	C
	mulhdu	r8, r27, v0	C 27
	ld	r27, 32(up)	C
	ld	r29, 16(rp)	C
	adde	r0, r0, r12	C 0 12
	adde	r24, r24, r10	C 24 10
	mulld	r9, r26, v0	C
	mulhdu	r10, r26, v0	C 26
	ld	r26, 40(up)	C
	ld	r30, 24(rp)	C
	mulld	r11, r27, v0	C
	mulhdu	r12, r27, v0	C 27
	ld	r27, 48(up)	C
	ld	r31, 32(rp)	C
	adde	r9, r9, r8	C 8 9
	adde	r11, r11, r10	C 10 11
	addze	r12, r12	C 12
	addc	r0, r0, r28	C 0 28
	std	r0, 8(rp)	C 0
	adde	r24, r24, r29	C 7 29
	std	r24, 16(rp)	C 7
	adde	r9, r9, r30	C 9 30
	std	r9, 24(rp)	C 9
	adde	r11, r11, r31	C 11 31
	std	r11, 32(rp)	C 11
	addi	up, up, 32	C
	addi	rp, rp, 32	C
	bdnz	L(lo_3)	C

	ALIGN(16)
L(end_3):
	mulld	r0, r26, v0
	mulhdu	r10, r26, v0
	ld	r28, 8(rp)
	nop
	mulld	r24, r27, v0
	mulhdu	r8, r27, v0
	ld	r29, 16(rp)
	nop
	adde	r0, r0, r12
	adde	r24, r24, r10
	addze	r8, r8
	addc	r0, r0, r28
	std	r0, 8(rp)
	adde	r24, r24, r29
	std	r24, 16(rp)
	addze	r8, r8
	std	r8, 24(rp)

	addic.	vn, vn, -1
	bne	L(outer_lo_3)
	b	L(ret)


	ALIGN(16)
L(b0):	ld	r27, 8(up)
	addi	up, up, 8
	mulld	r0, r26, v0
	mulhdu	r10, r26, v0
	mulld	r24, r27, v0
	mulhdu	r8, r27, v0
	addc	r24, r24, r10
	addze	r12, r8
	std	r0, 0(rp)
	std	r24, 8(rp)
	addi	rp, rp, 8
	ld	r26, 8(up)
	ld	r27, 16(up)
	bdz	L(end_m_0)

	ALIGN(16)
L(lo_m_0):
	mulld	r0, r26, v0
	mulhdu	r31, r26, v0
	ld	r26, 24(up)
	nop
	mulld	r24, r27, v0
	mulhdu	r8, r27, v0
	ld	r27, 32(up)
	nop
	adde	r0, r0, r12
	adde	r24, r24, r31
	mulld	r9, r26, v0
	mulhdu	r10, r26, v0
	ld	r26, 40(up)
	nop
	mulld	r11, r27, v0
	mulhdu	r12, r27, v0
	ld	r27, 48(up)
	std	r0, 8(rp)
	adde	r9, r9, r8
	std	r24, 16(rp)
	adde	r11, r11, r10
	std	r9, 24(rp)
	addi	up, up, 32
	std	r11, 32(rp)
	addi	rp, rp, 32
	bdnz	L(lo_m_0)

	ALIGN(16)
L(end_m_0):
	mulld	r0, r26, v0
	mulhdu	r31, r26, v0

	mulld	r24, r27, v0
	mulhdu	r8, r27, v0

	adde	r0, r0, r12
	adde	r24, r24, r31

	std	r0, 8(rp)
	addze	r8, r8
	std	r24, 16(rp)
	addic.	vn, vn, -1
	std	r8, 24(rp)
	nop
	beq	L(ret)

	ALIGN(16)
L(outer_lo_0):
	mtctr	un		C copy inner loop count into ctr
	addi	rp, outer_rp, 16
	addi	up, outer_up, 8
	addi	outer_rp, outer_rp, 8
	ld	v0, 0(vp)	C new v limb
	addi	vp, vp, 8
	ld	r26, -8(up)
	ld	r27, 0(up)
	ld	r28, -8(rp)
	ld	r29, 0(rp)
	nop
	nop
	mulld	r0, r26, v0
	mulhdu	r10, r26, v0
	mulld	r24, r27, v0
	mulhdu	r8, r27, v0
	addc	r24, r24, r10
	addze	r12, r8
	addc	r0, r0, r28
	std	r0, -8(rp)
	adde	r24, r24, r29
	std	r24, 0(rp)
	ld	r26, 8(up)
	ld	r27, 16(up)
	bdz	L(end_0)

	ALIGN(16)		C registers dying
L(lo_0):
	mulld	r0, r26, v0	C
	mulhdu	r10, r26, v0	C 26
	ld	r26, 24(up)	C
	ld	r28, 8(rp)	C
	mulld	r24, r27, v0	C
	mulhdu	r8, r27, v0	C 27
	ld	r27, 32(up)	C
	ld	r29, 16(rp)	C
	adde	r0, r0, r12	C 0 12
	adde	r24, r24, r10	C 24 10
	mulld	r9, r26, v0	C
	mulhdu	r10, r26, v0	C 26
	ld	r26, 40(up)	C
	ld	r30, 24(rp)	C
	mulld	r11, r27, v0	C
	mulhdu	r12, r27, v0	C 27
	ld	r27, 48(up)	C
	ld	r31, 32(rp)	C
	adde	r9, r9, r8	C 8 9
	adde	r11, r11, r10	C 10 11
	addze	r12, r12	C 12
	addc	r0, r0, r28	C 0 28
	std	r0, 8(rp)	C 0
	adde	r24, r24, r29	C 7 29
	std	r24, 16(rp)	C 7
	adde	r9, r9, r30	C 9 30
	std	r9, 24(rp)	C 9
	adde	r11, r11, r31	C 11 31
	std	r11, 32(rp)	C 11
	addi	up, up, 32	C
	addi	rp, rp, 32	C
	bdnz	L(lo_0)	C

	ALIGN(16)
L(end_0):
	mulld	r0, r26, v0
	mulhdu	r10, r26, v0
	ld	r28, 8(rp)
	nop
	mulld	r24, r27, v0
	mulhdu	r8, r27, v0
	ld	r29, 16(rp)
	nop
	adde	r0, r0, r12
	adde	r24, r24, r10
	addze	r8, r8
	addic.	vn, vn, -1
	addc	r0, r0, r28
	std	r0, 8(rp)
	adde	r24, r24, r29
	std	r24, 16(rp)
	addze	r8, r8
	std	r8, 24(rp)
	bne	L(outer_lo_0)
	b	L(ret)


	ALIGN(16)
L(b1):	ld	r27, 8(up)
	nop
	mulld	r0, r26, v0
	mulhdu	r31, r26, v0
	ld	r26, 16(up)
	mulld	r24, r27, v0
	mulhdu	r8, r27, v0
	mulld	r9, r26, v0
	mulhdu	r10, r26, v0
	addc	r24, r24, r31
	adde	r9, r9, r8
	addze	r12, r10
	std	r0, 0(rp)
	std	r24, 8(rp)
	std	r9, 16(rp)
	addi	up, up, 16
	addi	rp, rp, 16
	ld	r26, 8(up)
	ld	r27, 16(up)
	bdz	L(end_m_1)

	ALIGN(16)
L(lo_m_1):
	mulld	r0, r26, v0
	mulhdu	r31, r26, v0
	ld	r26, 24(up)
	nop
	mulld	r24, r27, v0
	mulhdu	r8, r27, v0
	ld	r27, 32(up)
	nop
	adde	r0, r0, r12
	adde	r24, r24, r31
	mulld	r9, r26, v0
	mulhdu	r10, r26, v0
	ld	r26, 40(up)
	nop
	mulld	r11, r27, v0
	mulhdu	r12, r27, v0
	ld	r27, 48(up)
	std	r0, 8(rp)
	adde	r9, r9, r8
	std	r24, 16(rp)
	adde	r11, r11, r10
	std	r9, 24(rp)
	addi	up, up, 32
	std	r11, 32(rp)
	addi	rp, rp, 32
	bdnz	L(lo_m_1)

	ALIGN(16)
L(end_m_1):
	mulld	r0, r26, v0
	mulhdu	r31, r26, v0

	mulld	r24, r27, v0
	mulhdu	r8, r27, v0

	adde	r0, r0, r12
	adde	r24, r24, r31

	std	r0, 8(rp)
	addze	r8, r8
	std	r24, 16(rp)
	addic.	vn, vn, -1
	std	r8, 24(rp)
	nop
	beq	L(ret)

	ALIGN(16)
L(outer_lo_1):
	mtctr	un		C copy inner loop count into ctr
	addi	rp, outer_rp, 24
	addi	up, outer_up, 16
	addi	outer_rp, outer_rp, 8
	ld	v0, 0(vp)	C new v limb
	addi	vp, vp, 8
	ld	r26, -16(up)
	ld	r27, -8(up)
	mulld	r0, r26, v0
	mulhdu	r31, r26, v0
	ld	r26, 0(up)
	ld	r28, -16(rp)
	mulld	r24, r27, v0
	mulhdu	r8, r27, v0
	ld	r29, -8(rp)
	ld	r30, 0(rp)
	mulld	r9, r26, v0
	mulhdu	r10, r26, v0
	addc	r24, r24, r31
	adde	r9, r9, r8
	addze	r12, r10
	addc	r0, r0, r28
	std	r0, -16(rp)
	adde	r24, r24, r29
	std	r24, -8(rp)
	adde	r9, r9, r30
	std	r9, 0(rp)
	ld	r26, 8(up)
	ld	r27, 16(up)
	bdz	L(end_1)

	ALIGN(16)		C registers dying
L(lo_1):
	mulld	r0, r26, v0	C
	mulhdu	r10, r26, v0	C 26
	ld	r26, 24(up)	C
	ld	r28, 8(rp)	C
	mulld	r24, r27, v0	C
	mulhdu	r8, r27, v0	C 27
	ld	r27, 32(up)	C
	ld	r29, 16(rp)	C
	adde	r0, r0, r12	C 0 12
	adde	r24, r24, r10	C 24 10
	mulld	r9, r26, v0	C
	mulhdu	r10, r26, v0	C 26
	ld	r26, 40(up)	C
	ld	r30, 24(rp)	C
	mulld	r11, r27, v0	C
	mulhdu	r12, r27, v0	C 27
	ld	r27, 48(up)	C
	ld	r31, 32(rp)	C
	adde	r9, r9, r8	C 8 9
	adde	r11, r11, r10	C 10 11
	addze	r12, r12	C 12
	addc	r0, r0, r28	C 0 28
	std	r0, 8(rp)	C 0
	adde	r24, r24, r29	C 7 29
	std	r24, 16(rp)	C 7
	adde	r9, r9, r30	C 9 30
	std	r9, 24(rp)	C 9
	adde	r11, r11, r31	C 11 31
	std	r11, 32(rp)	C 11
	addi	up, up, 32	C
	addi	rp, rp, 32	C
	bdnz	L(lo_1)	C

	ALIGN(16)
L(end_1):
	mulld	r0, r26, v0
	mulhdu	r10, r26, v0
	ld	r28, 8(rp)
	nop
	mulld	r24, r27, v0
	mulhdu	r8, r27, v0
	ld	r29, 16(rp)
	nop
	adde	r0, r0, r12
	adde	r24, r24, r10
	addze	r8, r8
	addic.	vn, vn, -1
	addc	r0, r0, r28
	std	r0, 8(rp)
	adde	r24, r24, r29
	std	r24, 16(rp)
	addze	r8, r8
	std	r8, 24(rp)
	bne	L(outer_lo_1)
	b	L(ret)


	ALIGN(16)
L(b2):	ld	r27, 8(up)
	addi	up, up, -8
	addi	rp, rp, -8
	li	r12, 0
	addic	r12, r12, 0

	ALIGN(16)
L(lo_m_2):
	mulld	r0, r26, v0
	mulhdu	r31, r26, v0
	ld	r26, 24(up)
	nop
	mulld	r24, r27, v0
	mulhdu	r8, r27, v0
	ld	r27, 32(up)
	nop
	adde	r0, r0, r12
	adde	r24, r24, r31
	mulld	r9, r26, v0
	mulhdu	r10, r26, v0
	ld	r26, 40(up)
	nop
	mulld	r11, r27, v0
	mulhdu	r12, r27, v0
	ld	r27, 48(up)
	std	r0, 8(rp)
	adde	r9, r9, r8
	std	r24, 16(rp)
	adde	r11, r11, r10
	std	r9, 24(rp)
	addi	up, up, 32
	std	r11, 32(rp)

	addi	rp, rp, 32
	bdnz	L(lo_m_2)

	ALIGN(16)
L(end_m_2):
	mulld	r0, r26, v0
	mulhdu	r31, r26, v0

	mulld	r24, r27, v0
	mulhdu	r8, r27, v0

	adde	r0, r0, r12
	adde	r24, r24, r31

	std	r0, 8(rp)
	addze	r8, r8
	std	r24, 16(rp)
	addic.	vn, vn, -1
	std	r8, 24(rp)
	nop
	beq	L(ret)

	ALIGN(16)
L(outer_lo_2):
	mtctr	un		C copy inner loop count into ctr
	addi	rp, outer_rp, 0
	addi	up, outer_up, -8
	addi	outer_rp, outer_rp, 8
	ld	v0, 0(vp)	C new v limb
	addi	vp, vp, 8
	ld	r26, 8(up)
	ld	r27, 16(up)
	li	r12, 0
	addic	r12, r12, 0

	ALIGN(16)		C registers dying
L(lo_2):
	mulld	r0, r26, v0	C
	mulhdu	r10, r26, v0	C 26
	ld	r26, 24(up)	C
	ld	r28, 8(rp)	C
	mulld	r24, r27, v0	C
	mulhdu	r8, r27, v0	C 27
	ld	r27, 32(up)	C
	ld	r29, 16(rp)	C
	adde	r0, r0, r12	C 0 12
	adde	r24, r24, r10	C 24 10
	mulld	r9, r26, v0	C
	mulhdu	r10, r26, v0	C 26
	ld	r26, 40(up)	C
	ld	r30, 24(rp)	C
	mulld	r11, r27, v0	C
	mulhdu	r12, r27, v0	C 27
	ld	r27, 48(up)	C
	ld	r31, 32(rp)	C
	adde	r9, r9, r8	C 8 9
	adde	r11, r11, r10	C 10 11
	addze	r12, r12	C 12
	addc	r0, r0, r28	C 0 28
	std	r0, 8(rp)	C 0
	adde	r24, r24, r29	C 7 29
	std	r24, 16(rp)	C 7
	adde	r9, r9, r30	C 9 30
	std	r9, 24(rp)	C 9
	adde	r11, r11, r31	C 11 31
	std	r11, 32(rp)	C 11
	addi	up, up, 32	C
	addi	rp, rp, 32	C
	bdnz	L(lo_2)	C

	ALIGN(16)
L(end_2):
	mulld	r0, r26, v0
	mulhdu	r10, r26, v0
	ld	r28, 8(rp)
	nop
	mulld	r24, r27, v0
	mulhdu	r8, r27, v0
	ld	r29, 16(rp)
	nop
	adde	r0, r0, r12
	adde	r24, r24, r10
	addze	r8, r8
	addic.	vn, vn, -1
	addc	r0, r0, r28
	std	r0, 8(rp)
	adde	r24, r24, r29
	std	r24, 16(rp)
	addze	r8, r8
	std	r8, 24(rp)
	bne	L(outer_lo_2)
	b	L(ret)


L(ret):	ld	r31, -8(r1)
	ld	r30, -16(r1)
	ld	r29, -24(r1)
	ld	r28, -32(r1)
	ld	r27, -40(r1)
	ld	r26, -48(r1)
	ld	r25, -56(r1)
	ld	r24, -64(r1)
	ld	r23, -72(r1)
	ld	r22, -80(r1)
	blr
EPILOGUE()
