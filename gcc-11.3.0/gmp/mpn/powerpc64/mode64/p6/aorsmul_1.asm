dnl  PowerPC-64 mpn_addmul_1 and mpn_submul_1 optimised for power6.

dnl  Copyright 1999-2001, 2003-2006, 2008, 2010, 2011 Free Software Foundation,
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

C               mpn_addmul_1    mpn_submul_1
C               cycles/limb     cycles/limb
C POWER3/PPC630     ?               ?
C POWER4/PPC970     ?               ?
C POWER5            ?               ?
C POWER6           12.25           12.8
C POWER7            ?               ?

C TODO
C  * Reduce register usage.
C  * Schedule function entry code.
C  * Unroll more.  8-way unrolling would bring us to 10 c/l, 16-way unrolling
C    would bring us to 9 c/l.
C  * Handle n = 1 and perhaps n = 2 separately, without saving any registers.

C INPUT PARAMETERS
define(`rp',  `r3')
define(`up',  `r4')
define(`n',   `r5')
define(`v0',  `r6')

ifdef(`OPERATION_addmul_1',`
  define(ADDSUBC,	adde)
  define(ADDSUB,	addc)
  define(func,		mpn_addmul_1)
  define(func_nc,	mpn_addmul_1c)	C FIXME: not really supported
  define(AM,		`$1')
  define(SM,		`')
  define(CLRRSC,	`addic	$1, r0, 0')
')
ifdef(`OPERATION_submul_1',`
  define(ADDSUBC,	subfe)
  define(ADDSUB,	subfc)
  define(func,		mpn_submul_1)
  define(func_nc,	mpn_submul_1c)	C FIXME: not really supported
  define(AM,		`')
  define(SM,		`$1')
  define(CLRRSC,	`subfc	$1, r0, r0')
')

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_submul_1)

ASM_START()
PROLOGUE(func)
	std	r31, -8(r1)
	std	r30, -16(r1)
	std	r29, -24(r1)
	std	r28, -32(r1)
	std	r27, -40(r1)

	rldicl.	r0, n, 0,62	C r0 = n & 3, set cr0
	cmpdi	cr6, r0, 2
	addi	n, n, 3		C compute count...
	srdi	n, n, 2		C ...for ctr
	mtctr	n		C copy loop count into ctr
	beq	cr0, L(b0)
	blt	cr6, L(b1)
	beq	cr6, L(b2)

L(b3):	ld	r8, 0(up)
	ld	r7, 8(up)
	ld	r27, 16(up)
	addi	up, up, 16
	addi	rp, rp, 16
	mulld	r5,  r8, v0
	mulhdu	r8,  r8, v0
	mulld	r9,  r7, v0
	mulhdu	r7,  r7, v0
	mulld	r11, r27, v0
	mulhdu	r27, r27, v0
	ld	r29, -16(rp)
	ld	r30, -8(rp)
	ld	r31, 0(rp)
	addc	r9, r9, r8
	adde	r11, r11, r7
	addze	r12, r27
	ADDSUB	r5, r5, r29
	b	L(l3)

L(b2):	ld	r7, 0(up)
	ld	r27, 8(up)
	addi	up, up, 8
	addi	rp, rp, 8
	mulld	r9,  r7, v0
	mulhdu	r7,  r7, v0
	mulld	r11, r27, v0
	mulhdu	r27, r27, v0
	ld	r30, -8(rp)
	ld	r31, 0(rp)
	addc	r11, r11, r7
	addze	r12, r27
	ADDSUB	r9, r9, r30
	b	L(l2)

L(b1):	ld	r27, 0(up)
	ld	r31, 0(rp)
	mulld	r11, r27, v0
	mulhdu	r12, r27, v0
	ADDSUB	r11, r11, r31
	b	L(l1)

L(b0):	addi	up, up, -8
	addi	rp, rp, -8
	CLRRSC(	r12)		C clear r12 and clr/set cy

	ALIGN(32)
L(top):
SM(`	subfe	r11, r0, r0')	C complement...
SM(`	addic	r11, r11, 1')	C ...carry flag
	ld	r10, 8(up)
	ld	r8, 16(up)
	ld	r7, 24(up)
	ld	r27, 32(up)
	addi	up, up, 32
	addi	rp, rp, 32
	mulld	r0,  r10, v0
	mulhdu	r10, r10, v0
	mulld	r5,  r8, v0
	mulhdu	r8,  r8, v0
	mulld	r9,  r7, v0
	mulhdu	r7,  r7, v0
	mulld	r11, r27, v0
	mulhdu	r27, r27, v0
	ld	r28, -24(rp)
	adde	r0, r0, r12
	ld	r29, -16(rp)
	adde	r5, r5, r10
	ld	r30, -8(rp)
	ld	r31, 0(rp)
	adde	r9, r9, r8
	adde	r11, r11, r7
	addze	r12, r27
	ADDSUB	r0, r0, r28
	std	r0, -24(rp)
	ADDSUBC	r5, r5, r29
L(l3):	std	r5, -16(rp)
	ADDSUBC	r9, r9, r30
L(l2):	std	r9, -8(rp)
	ADDSUBC	r11, r11, r31
L(l1):	std	r11, 0(rp)
	bdnz	L(top)

AM(`	addze	r3, r12')
SM(`	subfe	r11, r0, r0')		C complement...
	ld	r31, -8(r1)
SM(`	subf	r3, r11, r12')
	ld	r30, -16(r1)
	ld	r29, -24(r1)
	ld	r28, -32(r1)
	ld	r27, -40(r1)
	blr
EPILOGUE()
