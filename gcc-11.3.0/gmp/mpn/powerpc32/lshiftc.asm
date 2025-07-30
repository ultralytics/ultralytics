dnl  PowerPC-32 mpn_lshiftc.

dnl  Copyright 1995, 1998, 2000, 2002-2005, 2010 Free Software Foundation, Inc.

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

C                cycles/limb
C 603e:            ?
C 604e:            3.0
C 75x (G3):        3.0
C 7400,7410 (G4):  3.0
C 7445,7455 (G4+): 2.5
C 7447,7457 (G4+): 2.25
C power4/ppc970:   2.5
C power5:          2.5

C INPUT PARAMETERS
C rp	r3
C up	r4
C n	r5
C cnt	r6

ASM_START()
PROLOGUE(mpn_lshiftc)
	cmpwi	cr0, r5, 30	C more than 30 limbs?
	slwi	r0, r5, 2
	add	r4, r4, r0	C make r4 point at end of s1
	add	r7, r3, r0	C make r7 point at end of res
	bgt	L(BIG)		C branch if more than 12 limbs

	mtctr	r5		C copy size into CTR
	subfic	r8, r6, 32
	lwzu	r11, -4(r4)	C load first s1 limb
	srw	r3, r11, r8	C compute function return value
	bdz	L(end1)

L(oop):	lwzu	r10, -4(r4)
	slw	r9, r11, r6
	srw	r12, r10, r8
	nor	r9, r9, r12
	stwu	r9, -4(r7)
	bdz	L(end2)
	lwzu	r11, -4(r4)
	slw	r9, r10, r6
	srw	r12, r11, r8
	nor	r9, r9, r12
	stwu	r9, -4(r7)
	bdnz	L(oop)

L(end1):
	slw	r0, r11, r6
	nor	r0, r0, r0
	stw	r0, -4(r7)
	blr
L(end2):
	slw	r0, r10, r6
	nor	r0, r0, r0
	stw	r0, -4(r7)
	blr

L(BIG):
	stwu	r1, -48(r1)
	stmw	r24, 8(r1)	C save registers we are supposed to preserve
	lwzu	r9, -4(r4)
	subfic	r8, r6, 32
	srw	r3, r9, r8	C compute function return value
	slw	r0, r9, r6
	addi	r5, r5, -1

	andi.	r10, r5, 3	C count for spill loop
	beq	L(e)
	mtctr	r10
	lwzu	r28, -4(r4)
	bdz	L(xe0)

L(loop0):
	slw	r12, r28, r6
	srw	r24, r28, r8
	lwzu	r28, -4(r4)
	nor	r24, r0, r24
	stwu	r24, -4(r7)
	mr	r0, r12
	bdnz	L(loop0)	C taken at most once!

L(xe0):	slw	r12, r28, r6
	srw	r24, r28, r8
	nor	r24, r0, r24
	stwu	r24, -4(r7)
	mr	r0, r12

L(e):	srwi	r5, r5, 2	C count for unrolled loop
	addi	r5, r5, -1
	mtctr	r5
	lwz	r28, -4(r4)
	lwz	r29, -8(r4)
	lwz	r30, -12(r4)
	lwzu	r31, -16(r4)

L(loopU):
	slw	r9, r28, r6
	srw	r24, r28, r8
	lwz	r28, -4(r4)
	slw	r10, r29, r6
	srw	r25, r29, r8
	lwz	r29, -8(r4)
	slw	r11, r30, r6
	srw	r26, r30, r8
	lwz	r30, -12(r4)
	slw	r12, r31, r6
	srw	r27, r31, r8
	lwzu	r31, -16(r4)
	nor	r24, r0, r24
	stw	r24, -4(r7)
	nor	r25, r9, r25
	stw	r25, -8(r7)
	nor	r26, r10, r26
	stw	r26, -12(r7)
	nor	r27, r11, r27
	stwu	r27, -16(r7)
	mr	r0, r12
	bdnz	L(loopU)

	slw	r9, r28, r6
	srw	r24, r28, r8
	slw	r10, r29, r6
	srw	r25, r29, r8
	slw	r11, r30, r6
	srw	r26, r30, r8
	slw	r12, r31, r6
	srw	r27, r31, r8
	nor	r24, r0, r24
	stw	r24, -4(r7)
	nor	r25, r9, r25
	stw	r25, -8(r7)
	nor	r26, r10, r26
	stw	r26, -12(r7)
	nor	r27, r11, r27
	stw	r27, -16(r7)
	nor	r12, r12, r12
	stw	r12, -20(r7)
	lmw	r24, 8(r1)	C restore registers
	addi	r1, r1, 48
	blr
EPILOGUE()
