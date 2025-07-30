dnl  PowerPC-32 mpn_mul_1 -- Multiply a limb vector with a limb and store the
dnl  result in a second limb vector.

dnl  Copyright 1995, 1997, 2000, 2002, 2003, 2005 Free Software Foundation,
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

C                cycles/limb
C 603e:            ?
C 604e:            4.0
C 75x (G3):        4.5-11
C 7400,7410 (G4):  4.5-11
C 744x,745x (G4+): 6.0
C power4/ppc970:   6.0
C power5:          5.63

C INPUT PARAMETERS
C rp	r3
C up	r4
C n	r5
C vl	r6

ASM_START()
PROLOGUE(mpn_mul_1)
	mtctr	r5
	addi	r3,r3,-4	C adjust res_ptr, it's offset before it's used
	li	r12,0		C clear upper product reg
	addic	r0,r0,0		C clear cy
C Start software pipeline
	lwz	r8,0(r4)
	bdz	L(end3)
	lwzu	r9,4(r4)
	mullw	r11,r8,r6
	mulhwu	r0,r8,r6
	bdz	L(end1)
C Software pipelined main loop
L(loop):
	lwz	r8,4(r4)
	mullw	r10,r9,r6
	adde	r5,r11,r12
	mulhwu	r12,r9,r6
	stw	r5,4(r3)
	bdz	L(end2)
	lwzu	r9,8(r4)
	mullw	r11,r8,r6
	adde	r7,r10,r0
	mulhwu	r0,r8,r6
	stwu	r7,8(r3)
	bdnz	L(loop)
C Finish software pipeline
L(end1):
	mullw	r10,r9,r6
	adde	r5,r11,r12
	mulhwu	r12,r9,r6
	stw	r5,4(r3)
	adde	r7,r10,r0
	stwu	r7,8(r3)
	addze	r3,r12
	blr
L(end2):
	mullw	r11,r8,r6
	adde	r7,r10,r0
	mulhwu	r0,r8,r6
	stwu	r7,8(r3)
	adde	r5,r11,r12
	stw	r5,4(r3)
	addze	r3,r0
	blr
L(end3):
	mullw	r11,r8,r6
	stw	r11,4(r3)
	mulhwu	r3,r8,r6
	blr
EPILOGUE(mpn_mul_1)
