dnl  PowerPC-64 mpn_mod_1_1p

dnl  Copyright 2010, 2011 Free Software Foundation, Inc.

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

C                   cycles/limb
C POWER3/PPC630          ?
C POWER4/PPC970         17
C POWER5                16
C POWER6                30
C POWER7                10.2

C TODO
C  * Optimise, in particular the cps function.  This was compiler-generated and
C    then hand optimised.

C INPUT PARAMETERS
define(`ap',  `r3')
define(`n',   `r4')
define(`d',   `r5')
define(`cps', `r6')

ASM_START()

EXTERN_FUNC(mpn_invert_limb)

PROLOGUE(mpn_mod_1_1p)
	sldi	r10, r4, 3
	addi	r4, r4, -1
	add	r3, r3, r10
	ld	r0, 16(r6)		C B1modb
	ld	r12, 24(r6)		C B2modb
	ld	r9, -8(r3)
	ld	r10, -16(r3)
	mtctr	r4
	mulhdu	r8, r9, r0
	mulld	r7, r9, r0
	addc	r11, r7, r10
	addze	r9, r8
	bdz	L(end)

	ALIGN(16)
L(top):	ld	r4, -24(r3)
	addi	r3, r3, -8
	nop
	mulld	r10, r11, r0
	mulld	r8, r9, r12
	mulhdu	r11, r11, r0
	mulhdu	r9, r9, r12
	addc	r7, r10, r4
	addze	r10, r11
	addc	r11, r8, r7
	adde	r9, r9, r10
	bdnz	L(top)

L(end):
ifdef(`HAVE_LIMB_LITTLE_ENDIAN',
`	lwz	r0, 8(r6)',
`	lwz	r0, 12(r6)')
	ld	r3, 0(r6)
	cmpdi	cr7, r0, 0
	beq-	cr7, L(4)
	subfic	r10, r0, 64
	sld	r9, r9, r0
	srd	r10, r11, r10
	or	r9, r10, r9
L(4):	subfc	r10, r5, r9
	subfe	r10, r10, r10
	nand	r10, r10, r10
	sld	r11, r11, r0
	and	r10, r10, r5
	subf	r9, r10, r9
	mulhdu	r10, r9, r3
	mulld	r3, r9, r3
	addi	r9, r9, 1
	addc	r8, r3, r11
	adde	r3, r10, r9
	mulld	r3, r3, r5
	subf	r3, r3, r11
	cmpld	cr7, r8, r3
	bge	cr7, L(5)		C FIXME: Make branch-less
	add	r3, r3, r5
L(5):	cmpld	cr7, r3, r5
	bge-	cr7, L(10)
	srd	r3, r3, r0
	blr

L(10):	subf	r3, r5, r3
	srd	r3, r3, r0
	blr
EPILOGUE()

PROLOGUE(mpn_mod_1_1p_cps,toc)
	mflr	r0
	std	r29, -24(r1)
	std	r30, -16(r1)
	std	r31, -8(r1)
	cntlzd	r31, r4
	std	r0, 16(r1)
	extsw	r31, r31
	mr	r29, r3
	stdu	r1, -144(r1)
	sld	r30, r4, r31
	mr	r3, r30
	CALL(	mpn_invert_limb)
	cmpdi	cr7, r31, 0
	neg	r0, r30
	beq-	cr7, L(13)
	subfic	r11, r31, 64
	li	r0, 1
	neg	r9, r30
	srd	r11, r3, r11
	sld	r0, r0, r31
	or	r0, r11, r0
	mulld	r0, r0, r9
L(13):	mulhdu	r9, r0, r3
	mulld	r11, r0, r3
	add	r9, r0, r9
	nor	r9, r9, r9
	mulld	r9, r9, r30
	cmpld	cr7, r11, r9
	bge	cr7, L(14)
	add	r9, r9, r30
L(14):	addi	r1, r1, 144
	srd	r0, r0, r31
	std	r31, 8(r29)
	std	r3, 0(r29)
	std	r0, 16(r29)
	ld	r0, 16(r1)
	srd	r9, r9, r31
	ld	r30, -16(r1)
	ld	r31, -8(r1)
	std	r9, 24(r29)
	ld	r29, -24(r1)
	mtlr	r0
	blr
EPILOGUE()
