dnl  PowerPC-64 mpn_cnd_add_n/mpn_cnd_sub_n.

dnl  Copyright 1999-2001, 2003-2005, 2007, 2011, 2012 Free Software Foundation,
dnl  Inc.

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
C POWER4/PPC970          2.25
C POWER5                 ?
C POWER6                 3
C POWER7                 2

C INPUT PARAMETERS
define(`cnd',  `r3')
define(`rp',   `r4')
define(`up',   `r5')
define(`vp',   `r6')
define(`n',    `r7')

ifdef(`OPERATION_cnd_add_n',`
  define(ADDSUBC,	adde)
  define(ADDSUB,	addc)
  define(func,		mpn_cnd_add_n)
  define(GENRVAL,	`addi	r3, r3, 1')
  define(SETCBR,	`addic	r0, $1, -1')
  define(CLRCB,		`addic	r0, r0, 0')
')
ifdef(`OPERATION_cnd_sub_n',`
  define(ADDSUBC,	subfe)
  define(ADDSUB,	subfc)
  define(func,		mpn_cnd_sub_n)
  define(GENRVAL,	`neg	r3, r3')
  define(SETCBR,	`subfic	r0, $1, 0')
  define(CLRCB,		`addic	r0, r1, -1')
')

MULFUNC_PROLOGUE(mpn_cnd_add_n mpn_cnd_sub_n)

ASM_START()
PROLOGUE(func)
	std	r31, -8(r1)
	std	r30, -16(r1)
	std	r29, -24(r1)
	std	r28, -32(r1)
	std	r27, -40(r1)

	subfic	cnd, cnd, 0
	subfe	cnd, cnd, cnd

	rldicl.	r0, n, 0,62	C r0 = n & 3, set cr0
	cmpdi	cr6, r0, 2
	addi	n, n, 3	C compute count...
	srdi	n, n, 2	C ...for ctr
	mtctr	n		C copy count into ctr
	beq	cr0, L(b00)
	blt	cr6, L(b01)
	beq	cr6, L(b10)

L(b11):	ld	r8, 0(up)	C load s1 limb
	ld	r9, 0(vp)	C load s2 limb
	ld	r10, 8(up)	C load s1 limb
	ld	r11, 8(vp)	C load s2 limb
	ld	r12, 16(up)	C load s1 limb
	addi	up, up, 24
	ld	r0, 16(vp)	C load s2 limb
	addi	vp, vp, 24
	and	r9, r9, cnd
	and	r11, r11, cnd
	and	r0, r0, cnd
	ADDSUB	r29, r9, r8
	ADDSUBC	r30, r11, r10
	ADDSUBC	r31, r0, r12
	std	r29, 0(rp)
	std	r30, 8(rp)
	std	r31, 16(rp)
	addi	rp, rp, 24
	bdnz	L(go)
	b	L(ret)

L(b01):	ld	r12, 0(up)	C load s1 limb
	addi	up, up, 8
	ld	r0, 0(vp)	C load s2 limb
	addi	vp, vp, 8
	and	r0, r0, cnd
	ADDSUB	r31, r0, r12	C add
	std	r31, 0(rp)
	addi	rp, rp, 8
	bdnz	L(go)
	b	L(ret)

L(b10):	ld	r10, 0(up)	C load s1 limb
	ld	r11, 0(vp)	C load s2 limb
	ld	r12, 8(up)	C load s1 limb
	addi	up, up, 16
	ld	r0, 8(vp)	C load s2 limb
	addi	vp, vp, 16
	and	r11, r11, cnd
	and	r0, r0, cnd
	ADDSUB	r30, r11, r10	C add
	ADDSUBC	r31, r0, r12	C add
	std	r30, 0(rp)
	std	r31, 8(rp)
	addi	rp, rp, 16
	bdnz	L(go)
	b	L(ret)

L(b00):	CLRCB			C clear/set cy
L(go):	ld	r7, 0(up)	C load s1 limb
	ld	r27, 0(vp)	C load s2 limb
	ld	r8, 8(up)	C load s1 limb
	ld	r9, 8(vp)	C load s2 limb
	ld	r10, 16(up)	C load s1 limb
	ld	r11, 16(vp)	C load s2 limb
	ld	r12, 24(up)	C load s1 limb
	ld	r0, 24(vp)	C load s2 limb
	and	r27, r27, cnd
	and	r9, r9, cnd
	and	r11, r11, cnd
	and	r0, r0, cnd
	bdz	L(end)

	addi	up, up, 32
	addi	vp, vp, 32

L(top):	ADDSUBC	r28, r27, r7
	ld	r7, 0(up)	C load s1 limb
	ld	r27, 0(vp)	C load s2 limb
	ADDSUBC	r29, r9, r8
	ld	r8, 8(up)	C load s1 limb
	ld	r9, 8(vp)	C load s2 limb
	ADDSUBC	r30, r11, r10
	ld	r10, 16(up)	C load s1 limb
	ld	r11, 16(vp)	C load s2 limb
	ADDSUBC	r31, r0, r12
	ld	r12, 24(up)	C load s1 limb
	ld	r0, 24(vp)	C load s2 limb
	std	r28, 0(rp)
	addi	up, up, 32
	std	r29, 8(rp)
	addi	vp, vp, 32
	std	r30, 16(rp)
	std	r31, 24(rp)
	addi	rp, rp, 32
	and	r27, r27, cnd
	and	r9, r9, cnd
	and	r11, r11, cnd
	and	r0, r0, cnd
	bdnz	L(top)		C decrement ctr and loop back

L(end):	ADDSUBC	r28, r27, r7
	ADDSUBC	r29, r9, r8
	ADDSUBC	r30, r11, r10
	ADDSUBC	r31, r0, r12
	std	r28, 0(rp)
	std	r29, 8(rp)
	std	r30, 16(rp)
	std	r31, 24(rp)

L(ret):	ld	r31, -8(r1)
	ld	r30, -16(r1)
	ld	r29, -24(r1)
	ld	r28, -32(r1)
	ld	r27, -40(r1)

	subfe	r3, r0, r0	C -cy
	GENRVAL
	blr
EPILOGUE()
