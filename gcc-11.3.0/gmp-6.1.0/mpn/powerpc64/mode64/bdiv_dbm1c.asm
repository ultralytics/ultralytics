dnl  PPC64 mpn_bdiv_dbm1c.

dnl  Copyright 2008, 2010 Free Software Foundation, Inc.

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

C                 cycles/limb
C POWER3/PPC630       6-18
C POWER4/PPC970       8.25
C POWER5              8.5  fluctuating as function of n % 3
C POWER6             15
C POWER7              4.75

C TODO
C  * Nothing to do...

C INPUT PARAMETERS
define(`rp', `r3')
define(`up', `r4')
define(`n',  `r5')
define(`bd', `r6')
define(`cy', `r7')

ASM_START()
PROLOGUE(mpn_bdiv_dbm1c)
	ld	r0, 0(r4)

	rldicl.	r12, r5, 0,62
	cmpldi	cr6, r12, 2
	cmpldi	cr7, r5, 4
	addi	r5, r5, 1
	srwi	r5, r5, 2
	mtctr	r5
	beq	cr0, L(b00)
	blt	cr6, L(b01)
	beq	cr6, L(b10)

	ALIGN(16)
L(b11):	mulld	r5, r0, r6
	mulhdu	r12, r0, r6
	ld	r0, 8(r4)
	addi	r4, r4, -24
	addi	r3, r3, -24
	b	L(3)

	ALIGN(16)
L(b00):	mulld	r9, r0, r6
	mulhdu	r8, r0, r6
	addi	r4, r4, -16
	addi	r3, r3, -16
	b	L(0)

	ALIGN(16)
L(b01):	mulld	r5, r0, r6
	mulhdu	r12, r0, r6
	addi	r3, r3, -8
	ble	cr7, L(e1)
	ld	r0, 8(r4)
	addi	r4, r4, -8
	b	L(1)

	ALIGN(16)
L(b10):	mulld	r9, r0, r6
	mulhdu	r8, r0, r6
	ble	cr7, L(e2)

	ALIGN(16)
L(top):	subfc	r11, r9, r7
	ld	r10, 8(r4)
	ld	r0, 16(r4)
	subfe	r7, r8, r11
	std	r11, 0(r3)
	mulld	r5, r10, r6
	mulhdu	r12, r10, r6
L(1):	mulld	r9, r0, r6
	mulhdu	r8, r0, r6
	subfc	r11, r5, r7
	subfe	r7, r12, r11
	std	r11, 8(r3)
L(0):	subfc	r11, r9, r7
	ld	r10, 24(r4)
	ld	r0, 32(r4)
	subfe	r7, r8, r11
	std	r11, 16(r3)
	mulld	r5, r10, r6
	mulhdu	r12, r10, r6
L(3):	mulld	r9, r0, r6
	mulhdu	r8, r0, r6
	subfc	r11, r5, r7
	subfe	r7, r12, r11
	std	r11, 24(r3)
	addi	r4, r4, 32
	addi	r3, r3, 32
	bdnz	L(top)

L(e2):	ld	r10, 8(r4)
	mulld	r5, r10, r6
	mulhdu	r12, r10, r6
	subfc	r11, r9, r7
	subfe	r7, r8, r11
	std	r11, 0(r3)
L(e1):	subfc	r11, r5, r7
	std	r11, 8(r3)
	subfe	r3, r12, r11
	blr
EPILOGUE()
