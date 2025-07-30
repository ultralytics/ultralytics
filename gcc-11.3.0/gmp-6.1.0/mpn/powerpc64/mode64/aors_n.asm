dnl  PowerPC-64 mpn_add_n/mpn_sub_n -- mpn addition and subtraction.

dnl  Copyright 1999-2001, 2003-2005, 2007, 2011 Free Software Foundation, Inc.

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
C POWER3/PPC630          1.5
C POWER4/PPC970          2
C POWER5                 2
C POWER6                 2.63
C POWER7               2.25-2.87

C This code is a little bit slower for POWER3/PPC630 than the simple code used
C previously, but it is much faster for POWER4/PPC970.  The reason for the
C POWER3/PPC630 slowdown can be attributed to the saving and restoring of 4
C registers.

C INPUT PARAMETERS
C rp	r3
C up	r4
C vp	r5
C n	r6

ifdef(`OPERATION_add_n',`
  define(ADDSUBC,	adde)
  define(ADDSUB,	addc)
  define(func,		mpn_add_n)
  define(func_nc,	mpn_add_nc)
  define(GENRVAL,	`addi	r3, r3, 1')
  define(SETCBR,	`addic	r0, $1, -1')
  define(CLRCB,		`addic	r0, r0, 0')
')
ifdef(`OPERATION_sub_n',`
  define(ADDSUBC,	subfe)
  define(ADDSUB,	subfc)
  define(func,		mpn_sub_n)
  define(func_nc,	mpn_sub_nc)
  define(GENRVAL,	`neg	r3, r3')
  define(SETCBR,	`subfic	r0, $1, 0')
  define(CLRCB,		`addic	r0, r1, -1')
')

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc mpn_sub_n mpn_sub_nc)

ASM_START()
PROLOGUE(func_nc)
	SETCBR(r7)
	b	L(ent)
EPILOGUE()

PROLOGUE(func)
	CLRCB
L(ent):	std	r31, -8(r1)
	std	r30, -16(r1)
	std	r29, -24(r1)
	std	r28, -32(r1)

	rldicl.	r0, r6, 0,62	C r0 = n & 3, set cr0
	cmpdi	cr6, r0, 2
	addi	r6, r6, 3	C compute count...
	srdi	r6, r6, 2	C ...for ctr
	mtctr	r6		C copy count into ctr
	beq	cr0, L(b00)
	blt	cr6, L(b01)
	beq	cr6, L(b10)

L(b11):	ld	r8, 0(r4)	C load s1 limb
	ld	r9, 0(r5)	C load s2 limb
	ld	r10, 8(r4)	C load s1 limb
	ld	r11, 8(r5)	C load s2 limb
	ld	r12, 16(r4)	C load s1 limb
	addi	r4, r4, 24
	ld	r0, 16(r5)	C load s2 limb
	addi	r5, r5, 24
	ADDSUBC	r29, r9, r8
	ADDSUBC	r30, r11, r10
	ADDSUBC	r31, r0, r12
	std	r29, 0(r3)
	std	r30, 8(r3)
	std	r31, 16(r3)
	addi	r3, r3, 24
	bdnz	L(go)
	b	L(ret)

L(b01):	ld	r12, 0(r4)	C load s1 limb
	addi	r4, r4, 8
	ld	r0, 0(r5)	C load s2 limb
	addi	r5, r5, 8
	ADDSUBC	r31, r0, r12	C add
	std	r31, 0(r3)
	addi	r3, r3, 8
	bdnz	L(go)
	b	L(ret)

L(b10):	ld	r10, 0(r4)	C load s1 limb
	ld	r11, 0(r5)	C load s2 limb
	ld	r12, 8(r4)	C load s1 limb
	addi	r4, r4, 16
	ld	r0, 8(r5)	C load s2 limb
	addi	r5, r5, 16
	ADDSUBC	r30, r11, r10	C add
	ADDSUBC	r31, r0, r12	C add
	std	r30, 0(r3)
	std	r31, 8(r3)
	addi	r3, r3, 16
	bdnz	L(go)
	b	L(ret)

L(b00):	C INITCY		C clear/set cy
L(go):	ld	r6, 0(r4)	C load s1 limb
	ld	r7, 0(r5)	C load s2 limb
	ld	r8, 8(r4)	C load s1 limb
	ld	r9, 8(r5)	C load s2 limb
	ld	r10, 16(r4)	C load s1 limb
	ld	r11, 16(r5)	C load s2 limb
	ld	r12, 24(r4)	C load s1 limb
	ld	r0, 24(r5)	C load s2 limb
	bdz	L(end)

	addi	r4, r4, 32
	addi	r5, r5, 32

	ALIGN(16)
L(top):	ADDSUBC	r28, r7, r6
	ld	r6, 0(r4)	C load s1 limb
	ld	r7, 0(r5)	C load s2 limb
	ADDSUBC	r29, r9, r8
	ld	r8, 8(r4)	C load s1 limb
	ld	r9, 8(r5)	C load s2 limb
	ADDSUBC	r30, r11, r10
	ld	r10, 16(r4)	C load s1 limb
	ld	r11, 16(r5)	C load s2 limb
	ADDSUBC	r31, r0, r12
	ld	r12, 24(r4)	C load s1 limb
	ld	r0, 24(r5)	C load s2 limb
	std	r28, 0(r3)
	addi	r4, r4, 32
	std	r29, 8(r3)
	addi	r5, r5, 32
	std	r30, 16(r3)
	std	r31, 24(r3)
	addi	r3, r3, 32
	bdnz	L(top)		C decrement ctr and loop back

L(end):	ADDSUBC	r28, r7, r6
	ADDSUBC	r29, r9, r8
	ADDSUBC	r30, r11, r10
	ADDSUBC	r31, r0, r12
	std	r28, 0(r3)
	std	r29, 8(r3)
	std	r30, 16(r3)
	std	r31, 24(r3)

L(ret):	ld	r31, -8(r1)
	ld	r30, -16(r1)
	ld	r29, -24(r1)
	ld	r28, -32(r1)

	subfe	r3, r0, r0	C -cy
	GENRVAL
	blr
EPILOGUE()
