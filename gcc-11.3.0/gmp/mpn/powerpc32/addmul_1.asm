dnl  PowerPC-32 mpn_addmul_1 -- Multiply a limb vector with a limb and add the
dnl  result to a second limb vector.

dnl  Copyright 1995, 1997, 1998, 2000-2003, 2005 Free Software Foundation, Inc.

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
C 604e:            6.75
C 75x (G3):        8.7-14.3
C 7400,7410 (G4):  8.7-14.3
C 744x,745x (G4+): 9.5
C power4/ppc970:   6.25
C power5:          6.25

C INPUT PARAMETERS
C rp	r3
C up	r4
C n	r5
C vl	r6

C This is optimized for the PPC604.  It has not been tuned for other
C PowerPC processors.
C
C Loop Analysis for the 604:
C 12 mem insn
C 8 serializing insn
C 8 int multiply
C 25 int reg write
C 9 int ops (8 of which serialize)
C
C The multiply insns need 16 cycles/4limb.
C The integer register writes will need 13 cycles/4limb.
C All-in-all, it should be possible to get to 4 or 5 cycles/limb on PPC604,
C but that will require some clever FPNOPS and BNOPS for exact
C issue control.


ASM_START()
PROLOGUE(mpn_addmul_1)
	cmpwi	cr0,r5,9	C more than 9 limbs?
	bgt	cr0,L(big)	C branch if more than 9 limbs

	mtctr	r5
	lwz	r0,0(r4)
	mullw	r7,r0,r6
	mulhwu	r10,r0,r6
	lwz	r9,0(r3)
	addc	r8,r7,r9
	addi	r3,r3,-4
	bdz	L(end)
L(loop):
	lwzu	r0,4(r4)
	stwu	r8,4(r3)
	mullw	r8,r0,r6
	adde	r7,r8,r10
	mulhwu	r10,r0,r6
	lwz	r9,4(r3)
	addze	r10,r10
	addc	r8,r7,r9
	bdnz	L(loop)
L(end):	stw	r8,4(r3)
	addze	r3,r10
	blr

L(big):	stwu	r1,-16(r1)
	addi	r5,r5,-1
	stw	r30,8(r1)
	srwi	r0,r5,2
	stw	r31,12(r1)
	mtctr	r0

	lwz	r7,0(r4)
	mullw	r8,r7,r6
	mulhwu	r0,r7,r6
	lwz	r7,0(r3)
	addc	r8,r8,r7
	stw	r8,0(r3)

L(loopU):
	lwz	r7,4(r4)
	lwz	r12,8(r4)
	lwz	r30,12(r4)
	lwzu	r31,16(r4)
	mullw	r8,r7,r6
	mullw	r9,r12,r6
	mullw	r10,r30,r6
	mullw	r11,r31,r6
	adde	r8,r8,r0	C add cy_limb
	mulhwu	r0,r7,r6
	lwz	r7,4(r3)
	adde	r9,r9,r0
	mulhwu	r0,r12,r6
	lwz	r12,8(r3)
	adde	r10,r10,r0
	mulhwu	r0,r30,r6
	lwz	r30,12(r3)
	adde	r11,r11,r0
	mulhwu	r0,r31,r6
	lwz	r31,16(r3)
	addze	r0,r0		C new cy_limb
	addc	r8,r8,r7
	stw	r8,4(r3)
	adde	r9,r9,r12
	stw	r9,8(r3)
	adde	r10,r10,r30
	stw	r10,12(r3)
	adde	r11,r11,r31
	stwu	r11,16(r3)
	bdnz	L(loopU)

	andi.	r31,r5,3
	mtctr	r31
	beq	cr0,L(endx)

L(loopE):
	lwzu	r7,4(r4)
	mullw	r8,r7,r6
	adde	r8,r8,r0	C add cy_limb
	mulhwu	r0,r7,r6
	lwz	r7,4(r3)
	addze	r0,r0		C new cy_limb
	addc	r8,r8,r7
	stwu	r8,4(r3)
	bdnz	L(loopE)
L(endx):
	addze	r3,r0
	lwz	r30,8(r1)
	lwz	r31,12(r1)
	addi	r1,r1,16
	blr
EPILOGUE(mpn_addmul_1)
