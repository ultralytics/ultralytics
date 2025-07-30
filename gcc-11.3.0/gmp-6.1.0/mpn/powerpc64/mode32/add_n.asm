dnl  PowerPC-64/mode32 mpn_add_n -- Add two limb vectors of the same length > 0
dnl  and store sum in a third limb vector.

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
C POWER4/PPC970:     4.25

C INPUT PARAMETERS
C rp	r3
C up	r4
C vp	r5
C n	r6

ASM_START()
PROLOGUE(mpn_add_n)
	mtctr	r6		C copy size into CTR
	addic	r0, r0, 0	C clear cy
	ld	r8, 0(r4)	C load least significant s1 limb
	ld	r0, 0(r5)	C load least significant s2 limb
	addi	r3, r3, -8	C offset res_ptr, it's updated before it's used
	bdz	L(end)		C If done, skip loop

L(oop):	ld	r9, 8(r4)	C load s1 limb
	ld	r10, 8(r5)	C load s2 limb
	adde	r7, r0, r8	C add limbs with cy, set cy
	srdi	r6, r0, 32
	srdi	r11, r8, 32
	adde	r6, r6, r11	C add high limb parts, set cy
	std	r7, 8(r3)	C store result limb
	bdz	L(exit)		C decrement CTR and exit if done
	ldu	r8, 16(r4)	C load s1 limb and update s1_ptr
	ldu	r0, 16(r5)	C load s2 limb and update s2_ptr
	adde	r7, r10, r9	C add limbs with cy, set cy
	srdi	r6, r10, 32
	srdi	r11, r9, 32
	adde	r6, r6, r11	C add high limb parts, set cy
	stdu	r7, 16(r3)	C store result limb and update res_ptr
	bdnz	L(oop)		C decrement CTR and loop back

L(end):	adde	r7, r0, r8
	srdi	r6, r0, 32
	srdi	r11, r8, 32
	adde	r6, r6, r11	C add limbs with cy, set cy
	std	r7, 8(r3)	C store ultimate result limb
	li	r3, 0		C load cy into ...
	addze	r4, r3		C ... return value register
	blr
L(exit):	adde	r7, r10, r9
	srdi	r6, r10, 32
	srdi	r11, r9, 32
	adde	r6, r6, r11	C add limbs with cy, set cy
	std	r7, 16(r3)
	li	r3, 0		C load cy into ...
	addze	r4, r3		C ... return value register
	blr
EPILOGUE()
