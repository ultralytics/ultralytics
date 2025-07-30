dnl  PPC32 mpn_bdiv_dbm1c.

dnl  Copyright 2008 Free Software Foundation, Inc.

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
C 604e:            ?
C 75x (G3):        ?
C 7400,7410 (G4):  9.43
C 744x,745x (G4+): 6.28
C power4/ppc970:   ?
C power5:          ?

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
	lwz	r0, 0(r4)

	rlwinm.	r12, r5, 0,30,31
	cmplwi	cr6, r12, 2
	cmplwi	cr7, r5, 4
	addi	r5, r5, 1
	srwi	r5, r5, 2
	mtctr	r5
	beq	cr0, L(b00)
	blt	cr6, L(b01)
	beq	cr6, L(b10)

L(b11):	mullw	r5, r0, r6
	mulhwu	r12, r0, r6
	lwz	r0, 4(r4)
	addi	r4, r4, -12
	addi	r3, r3, -12
	b	L(3)

L(b00):	mullw	r9, r0, r6
	mulhwu	r8, r0, r6
	lwz	r0, 4(r4)
	addi	r4, r4, -8
	addi	r3, r3, -8
	b	L(0)

L(b01):	mullw	r5, r0, r6
	mulhwu	r12, r0, r6
	addi	r3, r3, -4
	ble	cr7, L(e1)
	lwz	r0, 4(r4)
	addi	r4, r4, -4
	b	L(1)

L(b10):	mullw	r9, r0, r6
	mulhwu	r8, r0, r6
	lwz	r0, 4(r4)
	ble	cr7, L(e2)

	ALIGN(16)
L(top):	mullw	r5, r0, r6
	mulhwu	r12, r0, r6
	subfc	r11, r9, r7
	lwz	r0, 8(r4)
	subfe	r7, r8, r11
	stw	r11, 0(r3)
L(1):	mullw	r9, r0, r6
	mulhwu	r8, r0, r6
	subfc	r11, r5, r7
	lwz	r0, 12(r4)
	subfe	r7, r12, r11
	stw	r11, 4(r3)
L(0):	mullw	r5, r0, r6
	mulhwu	r12, r0, r6
	subfc	r11, r9, r7
	lwz	r0, 16(r4)
	subfe	r7, r8, r11
	stw	r11, 8(r3)
L(3):	mullw	r9, r0, r6
	mulhwu	r8, r0, r6
	subfc	r11, r5, r7
	lwz	r0, 20(r4)
	subfe	r7, r12, r11
	stw	r11, 12(r3)
	addi	r4, r4, 16
	addi	r3, r3, 16
	bdnz	L(top)

L(e2):	mullw	r5, r0, r6
	mulhwu	r12, r0, r6
	subfc	r11, r9, r7
	subfe	r7, r8, r11
	stw	r11, 0(r3)
L(e1):	subfc	r11, r5, r7
	stw	r11, 4(r3)
	subfe	r3, r12, r11
	blr
EPILOGUE()
