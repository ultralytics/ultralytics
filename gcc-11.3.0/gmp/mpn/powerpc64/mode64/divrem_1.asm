dnl  PowerPC-64 mpn_divrem_1 -- Divide an mpn number by an unnormalized limb.

dnl  Copyright 2003-2005, 2007, 2008, 2010, 2012 Free Software Foundation, Inc.

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

C                           cycles/limb
C                       norm    unorm   frac
C POWER3/PPC630         16-34   16-34   ~11   outdated figures
C POWER4/PPC970          28      28      19
C POWER5                 29      29     ~19
C POWER6                 49      59     ~42
C POWER7                 24.5    23     ~14

C INPUT PARAMETERS
C qp  = r3
C fn  = r4
C up  = r5
C un  = r6
C d   = r7

C We use a not very predictable branch in the frac code, therefore the cycle
C count wobbles somewhat.  With the alternative branch-free code, things run
C considerably slower on POWER4/PPC970 and POWER5.

C Add preinv entry point.


ASM_START()

EXTERN_FUNC(mpn_invert_limb)

PROLOGUE(mpn_divrem_1,toc)

	mfcr	r12
	add.	r10, r6, r4
	std	r25, -56(r1)
	mr	r25, r4
	mflr	r0
	std	r26, -48(r1)
	mr	r26, r5
	std	r28, -32(r1)
	mr	r28, r6
	std	r29, -24(r1)
	mr	r29, r3
	li	r3, 0
	std	r30, -16(r1)
	mr	r30, r7
	std	r31, -8(r1)
	li	r31, 0
	std	r27, -40(r1)
	std	r0, 16(r1)
	stw	r12, 8(r1)
	stdu	r1, -176(r1)
	beq-	cr0, L(1)
	cmpdi	cr7, r7, 0
	sldi	r0, r10, 3
	add	r11, r0, r29
	addi	r29, r11, -8
	blt-	cr7, L(162)
	cmpdi	cr4, r6, 0
	beq+	cr4, L(71)
L(163):
	sldi	r9, r6, 3
	add	r9, r9, r5
	ld	r7, -8(r9)
	cmpld	cr7, r7, r30
	bge-	cr7, L(71)
	cmpdi	cr7, r10, 1
	li	r0, 0
	mr	r31, r7
	std	r0, -8(r11)
	addi	r29, r29, -8
	mr	r3, r7
	beq-	cr7, L(1)
	addi	r28, r6, -1
	cmpdi	cr4, r28, 0
L(71):
	cntlzd	r27, r30
	sld	r30, r30, r27
	sld	r31, r31, r27
	mr	r3, r30
	CALL(	mpn_invert_limb)
	beq-	cr4, L(110)
	sldi	r9, r28, 3
	addic.	r6, r28, -2
	add	r9, r9, r26
	subfic	r5, r27, 64
	ld	r8, -8(r9)
	srd	r0, r8, r5
	or	r31, r31, r0
	sld	r7, r8, r27
	blt-	cr0, L(154)
	addi	r28, r28, -1
	mtctr	r28
	sldi	r6, r6, 3
	ALIGN(16)
L(uloop):
	ldx	r8, r26, r6
	nop
	mulld	r0, r31, r3
	mulhdu	r10, r31, r3
	addi	r11, r31, 1
	srd	r9, r8, r5
	addi	r6, r6, -8
	or	r9, r7, r9
	addc	r0, r0, r9
	adde	r10, r10, r11
	mulld	r31, r10, r30
	subf	r31, r31, r9
	subfc	r0, r31, r0	C r <= ql
	subfe	r0, r0, r0	C r0 = -(r <= ql)
	and	r9, r30, r0
	add	r31, r31, r9
	add	r10, r0, r10	C qh -= (r >= ql)
	cmpld	cr7, r31, r30
	bge-	cr7, L(164)
L(123):
	std	r10, 0(r29)
	addi	r29, r29, -8
	sld	r7, r8, r27
	bdnz	L(uloop)
L(154):
	addi	r11, r31, 1
	nop
	mulld	r0, r31, r3
	mulhdu	r8, r31, r3
	addc	r0, r0, r7
	adde	r8, r8, r11
	mulld	r31, r8, r30
	subf	r31, r31, r7
	subfc	r0, r0, r31	C r >= ql
	subfe	r0, r0, r0	C r0 = -(r >= ql)
	not	r7, r0
	add	r8, r7, r8	C qh -= (r >= ql)
	andc	r0, r30, r0
	add	r31, r31, r0
	cmpld	cr7, r31, r30
	bge-	cr7, L(165)
L(134):
	std	r8, 0(r29)
	addi	r29, r29, -8
L(110):
	addic.	r0, r25, -1
	blt-	cr0, L(156)
	mtctr	r25
	neg	r9, r30
	ALIGN(16)
L(ufloop):
	addi	r11, r31, 1
	nop
	mulld	r0, r3, r31
	mulhdu	r10, r3, r31
	add	r10, r10, r11
	mulld	r31, r9, r10
ifelse(0,1,`
	subfc	r0, r0, r31
	subfe	r0, r0, r0	C r0 = -(r >= ql)
	not	r7, r0
	add	r10, r7, r10	C qh -= (r >= ql)
	andc	r0, r30, r0
	add	r31, r31, r0
',`
	cmpld	cr7, r31, r0
	blt	cr7, L(29)
	add	r31, r30, r31
	addi	r10, r10, -1
L(29):
')
	std	r10, 0(r29)
	addi	r29, r29, -8
	bdnz	L(ufloop)
L(156):
	srd	r3, r31, r27
L(1):
	addi	r1, r1, 176
	ld	r0, 16(r1)
	lwz	r12, 8(r1)
	mtlr	r0
	ld	r25, -56(r1)
	ld	r26, -48(r1)
	mtcrf	8, r12
	ld	r27, -40(r1)
	ld	r28, -32(r1)
	ld	r29, -24(r1)
	ld	r30, -16(r1)
	ld	r31, -8(r1)
	blr
L(162):
	cmpdi	cr7, r6, 0
	beq-	cr7, L(8)
	sldi	r9, r6, 3
	addi	r29, r29, -8
	add	r9, r9, r5
	addi	r28, r6, -1
	ld	r31, -8(r9)
	subfc	r9, r7, r31
	li	r9, 0
	adde	r9, r9, r9
	neg	r0, r9
	std	r9, -8(r11)
	and	r0, r0, r7
	subf	r31, r0, r31
L(8):
	mr	r3, r30
	CALL(	mpn_invert_limb)
	li	r27, 0
	addic.	r6, r28, -1
	blt-	cr0, L(110)
	mtctr	r28
	sldi	r6, r6, 3
	ALIGN(16)
L(nloop):
	addi	r11, r31, 1
	ldx	r8, r26, r6
	mulld	r0, r31, r3
	mulhdu	r10, r31, r3
	addi	r6, r6, -8
	addc	r0, r0, r8
	adde	r10, r10, r11
	mulld	r31, r10, r30
	subf	r31, r31, r8	C r = nl - qh * d
	subfc	r0, r31, r0	C r <= ql
	subfe	r0, r0, r0	C r0 = -(r <= ql)
	and	r9, r30, r0
	add	r31, r31, r9
	add	r10, r0, r10	C qh -= (r >= ql)
	cmpld	cr7, r31, r30
	bge-	cr7, L(167)
L(51):
	std	r10, 0(r29)
	addi	r29, r29, -8
	bdnz	L(nloop)
	b	L(110)

L(164):
	subf	r31, r30, r31
	addi	r10, r10, 1
	b	L(123)
L(167):
	subf	r31, r30, r31
	addi	r10, r10, 1
	b	L(51)
L(165):
	subf	r31, r30, r31
	addi	r8, r8, 1
	b	L(134)
EPILOGUE()
