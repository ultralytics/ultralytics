dnl  PowerPC-32 mpn_add_n/mpn_sub_n -- mpn addition and subtraction.

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
C POWER6                 2.78
C POWER7               2.15-2.87

C This code is based on powerpc64/aors_n.asm.

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
L(ent):	stwu	r1, -32(r1)
	rlwinm.	r0, r6, 0,30,31	C r0 = n & 3, set cr0
	cmpwi	cr6, r0, 2
	stw	r28, 8(r1)
	addi	r6, r6, 3	C compute count...
	stw	r29, 12(r1)
	srwi	r6, r6, 2	C ...for ctr
	stw	r30, 16(r1)
	mtctr	r6		C copy count into ctr
	stw	r31, 20(r1)
	beq	cr0, L(b00)
	blt	cr6, L(b01)
	beq	cr6, L(b10)

L(b11):	lwz	r8, 0(r4)	C load s1 limb
	lwz	r9, 0(r5)	C load s2 limb
	lwz	r10, 4(r4)	C load s1 limb
	lwz	r11, 4(r5)	C load s2 limb
	lwz	r12, 8(r4)	C load s1 limb
	addi	r4, r4, 12
	lwz	r0, 8(r5)	C load s2 limb
	addi	r5, r5, 12
	ADDSUBC	r29, r9, r8
	ADDSUBC	r30, r11, r10
	ADDSUBC	r31, r0, r12
	stw	r29, 0(r3)
	stw	r30, 4(r3)
	stw	r31, 8(r3)
	addi	r3, r3, 12
	bdnz	L(go)
	b	L(ret)

L(b01):	lwz	r12, 0(r4)	C load s1 limb
	addi	r4, r4, 4
	lwz	r0, 0(r5)	C load s2 limb
	addi	r5, r5, 4
	ADDSUBC	r31, r0, r12	C add
	stw	r31, 0(r3)
	addi	r3, r3, 4
	bdnz	L(go)
	b	L(ret)

L(b10):	lwz	r10, 0(r4)	C load s1 limb
	lwz	r11, 0(r5)	C load s2 limb
	lwz	r12, 4(r4)	C load s1 limb
	addi	r4, r4, 8
	lwz	r0, 4(r5)	C load s2 limb
	addi	r5, r5, 8
	ADDSUBC	r30, r11, r10	C add
	ADDSUBC	r31, r0, r12	C add
	stw	r30, 0(r3)
	stw	r31, 4(r3)
	addi	r3, r3, 8
	bdnz	L(go)
	b	L(ret)

L(b00):	C INITCY		C clear/set cy
L(go):	lwz	r6, 0(r4)	C load s1 limb
	lwz	r7, 0(r5)	C load s2 limb
	lwz	r8, 4(r4)	C load s1 limb
	lwz	r9, 4(r5)	C load s2 limb
	lwz	r10, 8(r4)	C load s1 limb
	lwz	r11, 8(r5)	C load s2 limb
	lwz	r12, 12(r4)	C load s1 limb
	lwz	r0, 12(r5)	C load s2 limb
	bdz	L(end)

	addi	r4, r4, 16
	addi	r5, r5, 16

	ALIGN(16)
L(top):	ADDSUBC	r28, r7, r6
	lwz	r6, 0(r4)	C load s1 limb
	lwz	r7, 0(r5)	C load s2 limb
	ADDSUBC	r29, r9, r8
	lwz	r8, 4(r4)	C load s1 limb
	lwz	r9, 4(r5)	C load s2 limb
	ADDSUBC	r30, r11, r10
	lwz	r10, 8(r4)	C load s1 limb
	lwz	r11, 8(r5)	C load s2 limb
	ADDSUBC	r31, r0, r12
	lwz	r12, 12(r4)	C load s1 limb
	lwz	r0, 12(r5)	C load s2 limb
	stw	r28, 0(r3)
	addi	r4, r4, 16
	stw	r29, 4(r3)
	addi	r5, r5, 16
	stw	r30, 8(r3)
	stw	r31, 12(r3)
	addi	r3, r3, 16
	bdnz	L(top)		C decrement ctr and loop back

L(end):	ADDSUBC	r28, r7, r6
	ADDSUBC	r29, r9, r8
	ADDSUBC	r30, r11, r10
	ADDSUBC	r31, r0, r12
	stw	r28, 0(r3)
	stw	r29, 4(r3)
	stw	r30, 8(r3)
	stw	r31, 12(r3)

L(ret):
	lwz	r28, 8(r1)
	lwz	r29, 12(r1)
	subfe	r3, r0, r0	C -cy
	lwz	r30, 16(r1)
	GENRVAL
	lwz	r31, 20(r1)
	addi	r1, r1, 32
	blr
EPILOGUE()
