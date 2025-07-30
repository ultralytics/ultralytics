dnl  PPC-64 mpn_divrem_2 -- Divide an mpn number by a normalized 2-limb number.

dnl  Copyright 2007, 2008 Free Software Foundation, Inc.

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

C                       cycles/limb
C                       norm    frac
C POWER3/PPC630
C POWER4/PPC970         ?       ?
C POWER5                37      ?
C POWER6                62      ?
C POWER6                30.5    ?

C INPUT PARAMETERS
C qp  = r3
C fn  = r4
C up  = r5
C un  = r6
C dp  = r7


ifdef(`DARWIN',,`
define(`r2',`r31')')		C FIXME!

ASM_START()

EXTERN_FUNC(mpn_invert_limb)

PROLOGUE(mpn_divrem_2,toc)
	mflr	r0
	std	r23, -72(r1)
	std	r24, -64(r1)
	std	r25, -56(r1)
	std	r26, -48(r1)
	std	r27, -40(r1)
	std	r28, -32(r1)
	std	r29, -24(r1)
	std	r30, -16(r1)
	std	r31, -8(r1)
	std	r0, 16(r1)
	stdu	r1, -192(r1)
	mr	r24, r3
	mr	r25, r4
	sldi	r0, r6, 3
	add	r26, r5, r0
	addi	r26, r26, -24
	ld	r30, 8(r7)
	ld	r28, 0(r7)
	ld	r29, 16(r26)
	ld	r31, 8(r26)

ifelse(0,1,`
	li	r23, 0
	cmpld	cr7, r29, r30
	blt	cr7, L(8)
	bgt	cr7, L(9)
	cmpld	cr0, r31, r28
	blt	cr0, L(8)
L(9):	subfc	r31, r28, r31
	subfe	r29, r30, r29
	li	r23, 1
',`
	li	r23, 0
	cmpld	cr7, r29, r30
	blt	cr7, L(8)
	mfcr	r0
	rlwinm	r0, r0, 30, 1
	subfc	r9, r28, r31
	addze.	r0, r0
	nop
	beq	cr0, L(8)
	subfc	r31, r28, r31
	subfe	r29, r30, r29
	li	r23, 1
')

L(8):
	add	r27, r25, r6
	addic.	r27, r27, -3
	blt	cr0, L(18)
	mr	r3, r30
	CALL(	mpn_invert_limb)
	mulld	r10, r3, r30
	mulhdu	r0, r3, r28
	addc	r8, r10, r28
	subfe	r11, r1, r1
	addc	r10, r8, r0
	addze.	r11, r11
	blt	cr0, L(91)
L(40):
	subfc	r10, r30, r10
	addme.	r11, r11
	addi	r3, r3, -1
	bge	cr0, L(40)
L(91):
	addi	r5, r27,  1
	mtctr	r5
	sldi	r0, r27, 3
	add	r24, r24, r0
	ALIGN(16)
L(loop):
	mulhdu	r8, r29, r3
	mulld	r6, r29, r3
	addc	r6, r6, r31
	adde	r8, r8, r29
	cmpd	cr7, r27, r25
	mulld	r0, r30, r8
	mulhdu	r11, r28, r8
	mulld	r10, r28, r8
	subf	r31, r0, r31
	li	r7, 0
	blt	cr7, L(60)
	ld	r7, 0(r26)
	addi	r26, r26, -8
	nop
L(60):	subfc	r7, r28, r7
	subfe	r31, r30, r31
	subfc	r7, r10, r7
	subfe	r4, r11, r31
	subfc	r9, r6, r4
	subfe	r9, r1, r1
	andc	r6, r28, r9
	andc	r0, r30, r9
	addc	r31, r7, r6
	adde	r29, r4, r0
	subf	r8, r9, r8
	cmpld	cr7, r29, r30
	bge-	cr7, L(fix)
L(bck):	std	r8, 0(r24)
	addi	r24, r24, -8
	addi	r27, r27, -1
	bdnz	L(loop)
L(18):
	std	r31, 8(r26)
	std	r29, 16(r26)
	mr	r3, r23
	addi	r1, r1, 192
	ld	r0, 16(r1)
	mtlr	r0
	ld	r23, -72(r1)
	ld	r24, -64(r1)
	ld	r25, -56(r1)
	ld	r26, -48(r1)
	ld	r27, -40(r1)
	ld	r28, -32(r1)
	ld	r29, -24(r1)
	ld	r30, -16(r1)
	ld	r31, -8(r1)
	blr
L(fix):
	mfcr	r0
	rlwinm	r0, r0, 30, 1
	subfc	r9, r28, r31
	addze.	r0, r0
	beq	cr0, L(bck)
	subfc	r31, r28, r31
	subfe	r29, r30, r29
	addi	r8, r8, 1
	b	L(bck)
EPILOGUE()
