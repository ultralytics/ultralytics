dnl  PowerPC-64 mpn_submul_1 -- Multiply a limb vector with a limb and subtract
dnl  the result from a second limb vector.

dnl  Copyright 1999-2001, 2003, 2005 Free Software Foundation, Inc.

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
C POWER3/PPC630:     ?
C POWER4/PPC970:     16

C INPUT PARAMETERS
C rp	r3
C up	r4
C n	r5
C v	r6,r7  or  r7,r8

ASM_START()
PROLOGUE(mpn_submul_1)

ifdef(`BROKEN_LONGLONG_PARAM',
`	rldimi	r8, r7, 32,0	C assemble vlimb from separate 32-bit arguments
	mr	r6, r8
',`
	rldimi	r7, r6, 32,0	C assemble vlimb from separate 32-bit arguments
	mr	r6, r7
')
	li	r7, 0		C cy_limb = 0
	mtctr	r5
	addic	r0, r0, 0
	addi	r3, r3, -8
	addi	r4, r4, -8

L(oop):	ldu	r0, 8(r4)
	mulld	r9, r0, r6
	adde	r12, r9, r7	C add old high limb and new low limb
	srdi	r5, r9, 32
	srdi	r11, r7, 32
	adde	r5, r5, r11	C add high limb parts, set cy
	mulhdu	r7, r0, r6
	addze	r7, r7
	ld	r10, 8(r3)
	subfc	r9, r12, r10
	srdi	r5, r12, 32
	srdi	r11, r10, 32
	subfe	r5, r5, r11	C subtract high limb parts, set cy
	stdu	r9, 8(r3)
	subfe	r11, r11, r11	C invert ...
	addic	r11, r11, 1	C ... carry
	bdnz	L(oop)

	addze	r4, r7
	srdi	r3, r4, 32
	blr
EPILOGUE()

