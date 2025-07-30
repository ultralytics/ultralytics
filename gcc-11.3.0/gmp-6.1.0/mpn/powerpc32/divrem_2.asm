dnl  PPC-32 mpn_divrem_2 -- Divide an mpn number by a normalized 2-limb number.

dnl  Copyright 2007, 2008, 2012 Free Software Foundation, Inc.

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

C		cycles/limb
C		norm	frac
C 7410		~36.5	~36.5
C 744x, 745x	 29	 29

C INPUT PARAMETERS
C qp  = r3
C fn  = r4
C up  = r5
C un  = r6
C d   = r7

C TODO
C  * Decrease register usage.
C  * Make sure mul operands and optimal for early-out.
C  * Check that things work well for a shared library build.
C  * Write an invert_limb, perhaps inline, perhaps as a private call.  Or at
C    least vastly improve the current __udiv_qrnnd_c based code.


ASM_START()
PROLOGUE(mpn_divrem_2)
	stwu	r1, -32(r1)
	slwi	r0, r6, 2
	add	r5, r5, r0
	stmw	r28, 8(r1)
	addi	r29, r5, -8		C up = up_param + un - 2
	lwz	r10, 4(r7)
	lwz	r12, 4(r29)
	addi	r8, r3, -12
	lwz	r7, 0(r7)
	cmplw	cr7, r12, r10
	lwz	r28, 0(r29)
	blt-	cr7, L(2)
	bgt+	cr7, L(4)
	cmplw	cr7, r28, r7
	blt-	cr7, L(2)
L(4):	subfc	r28, r7, r28
	subfe	r12, r10, r12
	li	r3, 1
	b	L(6)
L(2):	li	r3, 0

L(6):	add	r0, r4, r6
	addic.	r30, r0, -2
	ble-	cr0, L(ret)

	slwi	r9, r0, 2
	add	r8, r8, r9		C rp += un + fn
	mtctr	r30

C Compute di from d1
	srwi	r11, r10, 16
	nor	r0, r10, r10
	divwu	r31, r0, r11
	rlwinm	r5, r10, 0, 16, 31
	mullw	r9, r11, r31
	mullw	r6, r5, r31
	subf	r0, r9, r0
	slwi	r0, r0, 16
	ori	r0, r0, 65535
	cmplw	cr7, r0, r6
	bge-	cr7, L(9)
	add	r0, r0, r10
	cmplw	cr7, r0, r10
	cmplw	cr6, r6, r0
	addi	r31, r31, -1		C q1--
	crorc	28, 28, 25
	bc+	12, 28, L(9)
	addi	r31, r31, -1		C q1--
	add	r0, r0, r10
L(9):	subf	r0, r6, r0
	divwu	r6, r0, r11
	mullw	r9, r11, r6
	mullw	r11, r5, r6
	subf	r0, r9, r0
	slwi	r0, r0, 16
	ori	r0, r0, 65535
	cmplw	cr7, r0, r11
	bge-	cr7, L(13)
	add	r0, r0, r10
	cmplw	cr7, r0, r10
	cmplw	cr6, r11, r0
	addi	r6, r6, -1		C q0--
	crorc	28, 28, 25
	bc+	12, 28, L(13)
C	add	r0, r0, r10		C final remainder
	addi	r6, r6, -1		C q0--
L(13):	rlwimi	r6, r31, 16, 0, 15	C assemble final quotient

C Adjust di by including d0
	mullw	r9, r10, r6		C t0 = LO(di * d1)
	addc	r11, r9, r7
	subfe	r0, r1, r1
	mulhwu	r9, r6, r7		C s1 = HI(di * d0)
	addc	r9, r11, r9
	addze.	r0, r0
	blt	cr0, L(17)
L(18):	subfc	r9, r10, r9
	addi	r6, r6, -1
	addme.	r0, r0
	bge+	cr0, L(18)
L(17):

C r0  r3  r4  r5  r6  r7  r8  r9 r10 r11 r12 r28 r29 r30 r31
C     msl         di  d0  qp     d1          fn  up  un
L(loop):
	mullw	r0, r12, r6		C q0 = LO(n2 * di)
	cmpw	cr7, r30, r4
	addc	r31, r0, r28		C q0 += n1
	mulhwu	r9, r12, r6		C q  = HI(n2 * di)
	adde	r12, r9, r12		C q  += n2
	addi	r30, r30, -1
	mullw	r0, r10, r12		C d1 * q
	li	r9, 0
	subf	r0, r0, r28		C n1 -= d1 * q
	addi	r5, r12, 1
	ble-	cr7, L(23)
	lwzu	r9, -4(r29)
L(23):	mullw	r11, r12, r7		C t0 = LO(d0 * q)
	subfc	r28, r7, r9		C n0 -= d0
	subfe	r0, r10, r0		C n1 -= d1
	mulhwu	r12, r12, r7		C t1 = HI(d0 * q)
	subfc	r28, r11, r28		C n0 -= t0
	subfe	r12, r12, r0		C n1 -= t1
	cmplw	cr7, r12, r31
	blt+	cr7, L(24)
	addc	r28, r28, r7
	adde	r12, r12, r10
	addi	r5, r5, -1
L(24):	cmplw	cr7, r12, r10
	bge-	cr7, L(fix)
L(bck):	stw	r5, 0(r8)
	addi	r8, r8, -4
	bdnz	L(loop)

L(ret):	stw	r28, 0(r29)
	stw	r12, 4(r29)
	lmw	r28, 8(r1)
	addi	r1, r1, 32
	blr

L(fix):	cmplw	cr6, r28, r7
	bgt+	cr7, L(28)
	blt-	cr6, L(bck)
L(28):	subfc	r28, r7, r28
	subfe	r12, r10, r12
	addi	r5, r5, 1
	b	L(bck)
EPILOGUE()
