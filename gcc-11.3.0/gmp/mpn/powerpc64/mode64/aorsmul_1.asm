dnl  PowerPC-64 mpn_addmul_1 and mpn_submul_1.

dnl  Copyright 1999-2001, 2003-2006, 2010-2012 Free Software Foundation, Inc.

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

C                   mpn_addmul_1    mpn_submul_1
C                   cycles/limb     cycles/limb
C POWER3/PPC630		6-18		6-18
C POWER4/PPC970		 8		 8.3
C POWER5		 8		 8.25
C POWER6		16.25		16.75
C POWER7		 3.77		 4.9

C TODO
C  * Try to reduce the number of needed live registers
C  * Add support for _1c entry points

C INPUT PARAMETERS
define(`rp', `r3')
define(`up', `r4')
define(`n',  `r5')
define(`vl', `r6')

ifdef(`OPERATION_addmul_1',`
  define(ADDSUBC,	adde)
  define(ADDSUB,	addc)
  define(func,		mpn_addmul_1)
  define(func_nc,	mpn_addmul_1c)	C FIXME: not really supported
  define(SM,		`')
')
ifdef(`OPERATION_submul_1',`
  define(ADDSUBC,	subfe)
  define(ADDSUB,	subfc)
  define(func,		mpn_submul_1)
  define(func_nc,	mpn_submul_1c)	C FIXME: not really supported
  define(SM,		`$1')
')

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_submul_1)

ASM_START()
PROLOGUE(func)
	std	r31, -8(r1)
	rldicl.	r0, n, 0,62	C r0 = n & 3, set cr0
	std	r30, -16(r1)
	cmpdi	cr6, r0, 2
	std	r29, -24(r1)
	addi	n, n, 3		C compute count...
	std	r28, -32(r1)
	srdi	n, n, 2		C ...for ctr
	std	r27, -40(r1)
	mtctr	n		C copy count into ctr
	beq	cr0, L(b00)
	blt	cr6, L(b01)
	beq	cr6, L(b10)

L(b11):	ld	r9, 0(up)
	ld	r28, 0(rp)
	mulld	r0, r9, r6
	mulhdu	r12, r9, r6
	ADDSUB	r0, r0, r28
	std	r0, 0(rp)
	addi	rp, rp, 8
	ld	r9, 8(up)
	ld	r27, 16(up)
	addi	up, up, 24
SM(`	subfe	r11, r11, r11 ')
	b	L(bot)

	ALIGN(16)
L(b00):	ld	r9, 0(up)
	ld	r27, 8(up)
	ld	r28, 0(rp)
	ld	r29, 8(rp)
	mulld	r0, r9, r6
	mulhdu	r5, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	addc	r7, r7, r5
	addze	r12, r8
	ADDSUB	r0, r0, r28
	std	r0, 0(rp)
	ADDSUBC	r7, r7, r29
	std	r7, 8(rp)
	addi	rp, rp, 16
	ld	r9, 16(up)
	ld	r27, 24(up)
	addi	up, up, 32
SM(`	subfe	r11, r11, r11 ')
	b	L(bot)

	ALIGN(16)
L(b01):	bdnz	L(gt1)
	ld	r9, 0(up)
	ld	r11, 0(rp)
	mulld	r0, r9, r6
	mulhdu	r8, r9, r6
	ADDSUB	r0, r0, r11
	std	r0, 0(rp)
SM(`	subfe	r11, r11, r11 ')
SM(`	addic	r11, r11, 1 ')
	addze	r3, r8
	blr
L(gt1):	ld	r9, 0(up)
	ld	r27, 8(up)
	mulld	r0, r9, r6
	mulhdu	r5, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r9, 16(up)
	ld	r28, 0(rp)
	ld	r29, 8(rp)
	ld	r30, 16(rp)
	mulld	r11, r9, r6
	mulhdu	r10, r9, r6
	addc	r7, r7, r5
	adde	r11, r11, r8
	addze	r12, r10
	ADDSUB	r0, r0, r28
	std	r0, 0(rp)
	ADDSUBC	r7, r7, r29
	std	r7, 8(rp)
	ADDSUBC	r11, r11, r30
	std	r11, 16(rp)
	addi	rp, rp, 24
	ld	r9, 24(up)
	ld	r27, 32(up)
	addi	up, up, 40
SM(`	subfe	r11, r11, r11 ')
	b	L(bot)

L(b10):	addic	r0, r0, 0
	li	r12, 0		C cy_limb = 0
	ld	r9, 0(up)
	ld	r27, 8(up)
	bdz	L(end)
	addi	up, up, 16

	ALIGN(16)
L(top):	mulld	r0, r9, r6
	mulhdu	r5, r9, r6	C 9
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6	C 27
	ld	r9, 0(up)
	ld	r28, 0(rp)
	ld	r27, 8(up)
	ld	r29, 8(rp)
	adde	r0, r0, r12	C 0 12
	adde	r7, r7, r5	C 5 7
	mulld	r5, r9, r6
	mulhdu	r10, r9, r6	C 9
	mulld	r11, r27, r6
	mulhdu	r12, r27, r6	C 27
	ld	r9, 16(up)
	ld	r30, 16(rp)
	ld	r27, 24(up)
	ld	r31, 24(rp)
	adde	r5, r5, r8	C 8 5
	adde	r11, r11, r10	C 10 11
	addze	r12, r12	C 12
	ADDSUB	r0, r0, r28	C 0 28
	std	r0, 0(rp)	C 0
	ADDSUBC	r7, r7, r29	C 7 29
	std	r7, 8(rp)	C 7
	ADDSUBC	r5, r5, r30	C 5 30
	std	r5, 16(rp)	C 5
	ADDSUBC	r11, r11, r31	C 11 31
	std	r11, 24(rp)	C 11
	addi	up, up, 32
SM(`	subfe	r11, r11, r11 ')
	addi	rp, rp, 32
L(bot):
SM(`	addic	r11, r11, 1 ')
	bdnz	L(top)

L(end):	mulld	r0, r9, r6
	mulhdu	r5, r9, r6
	mulld	r7, r27, r6
	mulhdu	r8, r27, r6
	ld	r28, 0(rp)
	ld	r29, 8(rp)
	adde	r0, r0, r12
	adde	r7, r7, r5
	addze	r8, r8
	ADDSUB	r0, r0, r28
	std	r0, 0(rp)
	ADDSUBC	r7, r7, r29
	std	r7, 8(rp)
SM(`	subfe	r11, r11, r11 ')
SM(`	addic	r11, r11, 1 ')
	addze	r3, r8
	ld	r31, -8(r1)
	ld	r30, -16(r1)
	ld	r29, -24(r1)
	ld	r28, -32(r1)
	ld	r27, -40(r1)
	blr
EPILOGUE()
