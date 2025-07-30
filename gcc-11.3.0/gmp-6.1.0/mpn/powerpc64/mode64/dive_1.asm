dnl  PowerPC-64 mpn_divexact_1 -- mpn by limb exact division.

dnl  Copyright 2006, 2010 Free Software Foundation, Inc.

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
C                       norm    unorm
C POWER3/PPC630        13-19
C POWER4/PPC970         16
C POWER5                16      16
C POWER6                37      46
C POWER7                12      12

C TODO
C  * Check if n=1 code is really an improvement.  It probably isn't.
C  * Make more similar to mode1o.asm.

C INPUT PARAMETERS
define(`rp', `r3')
define(`up', `r4')
define(`n',  `r5')
define(`d',  `r6')


ASM_START()

EXTERN(binvert_limb_table)

PROLOGUE(mpn_divexact_1,toc)
	addic.	n, n, -1
	ld	r12, 0(up)
	bne	cr0, L(2)
	divdu	r0, r12, d
	std	r0, 0(rp)
	blr
L(2):
	rldicl.	r0, d, 0, 63
	li	r10, 0
	bne	cr0, L(7)
	neg	r0, d
	and	r0, d, r0
	cntlzd	r0, r0
	subfic	r0, r0, 63
	rldicl	r10, r0, 0, 32
	srd	d, d, r0
L(7):
	mtctr	n
	LEA(	r5, binvert_limb_table)
	rldicl	r11, d, 63, 57
	lbzx	r0, r5, r11
	mulld	r9, r0, r0
	sldi	r0, r0, 1
	mulld	r9, d, r9
	subf	r0, r9, r0
	mulld	r5, r0, r0
	sldi	r0, r0, 1
	mulld	r5, d, r5
	subf	r0, r5, r0
	mulld	r9, r0, r0
	sldi	r0, r0, 1
	mulld	r9, d, r9
	subf	r7, r9, r0		C r7 = 1/d mod 2^64
	bne	cr0, L(norm)
	subfic	r8, r10, 64		C set carry as side effect
	li	r5, 0
	srd	r11, r12, r10

	ALIGN(16)
L(loop0):
	ld	r12, 8(up)
	nop
	addi	up, up, 8
	sld	r0, r12, r8
	or	r11, r11, r0
	subfe	r9, r5, r11
	srd	r11, r12, r10
	mulld	r0, r7, r9
	mulhdu	r5, r0, d
	std	r0, 0(rp)
	addi	rp, rp, 8
	bdnz	L(loop0)

	subfe	r0, r5, r11
	mulld	r0, r7, r0
	std	r0, 0(rp)
	blr

	ALIGN(16)
L(norm):
	mulld	r11, r12, r7
	mulhdu	r5, r11, d
	std	r11, 0(rp)
	ALIGN(16)
L(loop1):
	ld	r9, 8(up)
	addi	up, up, 8
	subfe	r5, r5, r9
	mulld	r11, r7, r5
	mulhdu	r5, r11, d	C result not used
	std	r11, 8(rp)
	addi	rp, rp, 8
	bdnz	L(loop1)
	blr
EPILOGUE()
ASM_END()
