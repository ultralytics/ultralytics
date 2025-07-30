dnl  PowerPC-64 mpn_lshift -- rp[] = up[] << cnt

dnl  Copyright 2003, 2005, 2010, 2011 Free Software Foundation, Inc.

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
C POWER4/PPC970          ?
C POWER5                 2.25
C POWER6                 9.75
C POWER7                 2.15

C TODO
C  * Try to reduce the number of needed live registers
C  * Micro-optimise header code
C  * Keep in synch with rshift.asm and lshiftc.asm

C INPUT PARAMETERS
define(`rp',  `r3')
define(`up',  `r4')
define(`n',   `r5')
define(`cnt', `r6')

define(`tnc',`r0')
define(`u0',`r30')
define(`u1',`r31')
define(`retval',`r5')

ASM_START()
PROLOGUE(mpn_lshift)
	std	r31, -8(r1)
	std	r30, -16(r1)
	subfic	tnc, cnt, 64
	sldi	r7, n, 3	C byte count corresponding to n
	add	up, up, r7	C up = up + n
	add	rp, rp, r7	C rp = rp + n
	rldicl.	r30, n, 0,62	C r30 = n & 3, set cr0
	cmpdi	cr6, r30, 2
	addi	r31, n, 3	C compute count...
	ld	r10, -8(up)	C load 1st limb for b00...b11
	srd	retval, r10, tnc
ifdef(`HAVE_ABI_mode32',
`	rldicl	r31, r31, 62,34',	C ...branch count
`	srdi	r31, r31, 2')	C ...for ctr
	mtctr	r31		C copy count into ctr
	beq	cr0, L(b00)
	blt	cr6, L(b01)
	ld	r11, -16(up)	C load 2nd limb for b10 and b11
	beq	cr6, L(b10)

	ALIGN(16)
L(b11):	sld	r8, r10, cnt
	srd	r9, r11, tnc
	ld	u1, -24(up)
	addi	up, up, -24
	sld	r12, r11, cnt
	srd	r7, u1, tnc
	addi	rp, rp, 16
	bdnz	L(gt3)

	or	r11, r8, r9
	sld	r8, u1, cnt
	b	L(cj3)

	ALIGN(16)
L(gt3):	ld	u0, -8(up)
	or	r11, r8, r9
	sld	r8, u1, cnt
	srd	r9, u0, tnc
	ld	u1, -16(up)
	or	r10, r12, r7
	b	L(L11)

	ALIGN(32)
L(b10):	sld	r12, r10, cnt
	addi	rp, rp, 24
	srd	r7, r11, tnc
	bdnz	L(gt2)

	sld	r8, r11, cnt
	or	r10, r12, r7
	b	L(cj2)

L(gt2):	ld	u0, -24(up)
	sld	r8, r11, cnt
	srd	r9, u0, tnc
	ld	u1, -32(up)
	or	r10, r12, r7
	sld	r12, u0, cnt
	srd	r7, u1, tnc
	ld	u0, -40(up)
	or	r11, r8, r9
	addi	up, up, -16
	b	L(L10)

	ALIGN(16)
L(b00):	ld	u1, -16(up)
	sld	r12, r10, cnt
	srd	r7, u1, tnc
	ld	u0, -24(up)
	sld	r8, u1, cnt
	srd	r9, u0, tnc
	ld	u1, -32(up)
	or	r10, r12, r7
	sld	r12, u0, cnt
	srd	r7, u1, tnc
	addi	rp, rp, 8
	bdz	L(cj4)

L(gt4):	addi	up, up, -32
	ld	u0, -8(up)
	or	r11, r8, r9
	b	L(L00)

	ALIGN(16)
L(b01):	bdnz	L(gt1)
	sld	r8, r10, cnt
	std	r8, -8(rp)
	b	L(ret)

L(gt1):	ld	u0, -16(up)
	sld	r8, r10, cnt
	srd	r9, u0, tnc
	ld	u1, -24(up)
	sld	r12, u0, cnt
	srd	r7, u1, tnc
	ld	u0, -32(up)
	or	r11, r8, r9
	sld	r8, u1, cnt
	srd	r9, u0, tnc
	ld	u1, -40(up)
	addi	up, up, -40
	or	r10, r12, r7
	bdz	L(end)

	ALIGN(32)
L(top):	sld	r12, u0, cnt
	srd	r7, u1, tnc
	ld	u0, -8(up)
	std	r11, -8(rp)
	or	r11, r8, r9
L(L00):	sld	r8, u1, cnt
	srd	r9, u0, tnc
	ld	u1, -16(up)
	std	r10, -16(rp)
	or	r10, r12, r7
L(L11):	sld	r12, u0, cnt
	srd	r7, u1, tnc
	ld	u0, -24(up)
	std	r11, -24(rp)
	or	r11, r8, r9
L(L10):	sld	r8, u1, cnt
	srd	r9, u0, tnc
	ld	u1, -32(up)
	addi	up, up, -32
	std	r10, -32(rp)
	addi	rp, rp, -32
	or	r10, r12, r7
	bdnz	L(top)

	ALIGN(32)
L(end):	sld	r12, u0, cnt
	srd	r7, u1, tnc
	std	r11, -8(rp)
L(cj4):	or	r11, r8, r9
	sld	r8, u1, cnt
	std	r10, -16(rp)
L(cj3):	or	r10, r12, r7
	std	r11, -24(rp)
L(cj2):	std	r10, -32(rp)
	std	r8, -40(rp)

L(ret):	ld	r31, -8(r1)
	ld	r30, -16(r1)
ifdef(`HAVE_ABI_mode32',
`	srdi	r3, retval, 32
	mr	r4, retval
',`	mr	r3, retval')
	blr
EPILOGUE()
