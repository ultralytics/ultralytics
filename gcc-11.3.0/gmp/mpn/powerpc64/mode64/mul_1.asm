dnl  PowerPC-64 mpn_mul_1 -- Multiply a limb vector with a limb and store
dnl  the result in a second limb vector.

dnl  Copyright 1999-2001, 2003-2006, 2010 Free Software Foundation, Inc.

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

C               cycles/limb
C POWER3/PPC630     6-18
C POWER4/PPC970     7.25?  not updated for last file revision
C POWER5            7.25
C POWER6           14
C POWER7            2.9

C TODO
C  * Try to reduce the number of needed live registers (at least r5 and r10
C    could be combined)
C  * Optimize feed-in code, for speed and size.
C  * Clean up r12/r7 usage in feed-in code.

C INPUT PARAMETERS
define(`rp', `r3')
define(`up', `r4')
define(`n', `r5')
define(`vl', `r6')

ASM_START()
PROLOGUE(mpn_mul_1c)
	std	r27, -40(r1)
	std	r26, -48(r1)
	mr	r12, r7
	b	L(ent)
EPILOGUE()
PROLOGUE(mpn_mul_1)
	std	r27, -40(r1)
	std	r26, -48(r1)
	li	r12, 0		C cy_limb = 0
L(ent):	ld	r26, 0(up)

	rldicl.	r0, n, 0,62	C r0 = n & 3, set cr0
	cmpdi	cr6, r0, 2
	addic	n, n, 3		C compute count...
	srdi	n, n, 2		C ...for ctr
	mtctr	n		C copy count into ctr
	beq	cr0, L(b00)
	blt	cr6, L(b01)
	beq	cr6, L(b10)

L(b11):	mr	r7, r12
	mulld	r0, r26, r6
	mulhdu	r12, r26, r6
	addi	up, up, 8
	addc	r0, r0, r7
	std	r0, 0(rp)
	addi	rp, rp, 8
	b	L(fic)

L(b00):	ld	r27, 8(up)
	addi	up, up, 16
	mulld	r0, r26, r6
	mulhdu	r5, r26, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	addc	r0, r0, r12
	adde	r7, r7, r5
	addze	r12, r8
	std	r0, 0(rp)
	std	r7, 8(rp)
	addi	rp, rp, 16
	b	L(fic)

	nop			C alignment
L(b01):	bdnz	L(gt1)
	mulld	r0, r26, r6
	mulhdu	r8, r26, r6
	addc	r0, r0, r12
	std	r0, 0(rp)
	b	L(ret)
L(gt1):	ld	r27, 8(up)
	nop
	mulld	r0, r26, r6
	mulhdu	r5, r26, r6
	ld	r26, 16(up)
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	mulld	r9, r26, r6
	mulhdu	r10, r26, r6
	addc	r0, r0, r12
	adde	r7, r7, r5
	adde	r9, r9, r8
	addze	r12, r10
	std	r0, 0(rp)
	std	r7, 8(rp)
	std	r9, 16(rp)
	addi	up, up, 24
	addi	rp, rp, 24
	b	L(fic)

	nop
L(fic):	ld	r26, 0(up)
L(b10):	ld	r27, 8(up)
	addi	up, up, 16
	bdz	L(end)

L(top):	mulld	r0, r26, r6
	mulhdu	r5, r26, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r26, 0(up)
	ld	r27, 8(up)
	adde	r0, r0, r12
	adde	r7, r7, r5
	mulld	r9, r26, r6
	mulhdu	r10, r26, r6
	mulld	r11, r27, r6
	mulhdu	r12, r27, r6
	ld	r26, 16(up)
	ld	r27, 24(up)
	std	r0, 0(rp)
	adde	r9, r9, r8
	std	r7, 8(rp)
	adde	r11, r11, r10
	std	r9, 16(rp)
	addi	up, up, 32
	std	r11, 24(rp)

	addi	rp, rp, 32
	bdnz	L(top)

L(end):	mulld	r0, r26, r6
	mulhdu	r5, r26, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	adde	r0, r0, r12
	adde	r7, r7, r5
	std	r0, 0(rp)
	std	r7, 8(rp)
L(ret):	addze	r3, r8
	ld	r27, -40(r1)
	ld	r26, -48(r1)
	blr
EPILOGUE()
