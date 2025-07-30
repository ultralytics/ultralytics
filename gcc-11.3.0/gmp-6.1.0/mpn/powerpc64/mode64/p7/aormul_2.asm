dnl  PowerPC-64 mpn_mul_2 and mpn_addmul_2.

dnl  Copyright 2013 Free Software Foundation, Inc.

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

C                    cycles/limb    cycles/limb
C			mul_2         addmul_2
C POWER3/PPC630		 ?		 ?
C POWER4/PPC970		 ?		 ?
C POWER5		 ?		 ?
C POWER6		 ?		 ?
C POWER7-SMT4		 3		 3
C POWER7-SMT2		 ?		 ?
C POWER7-SMT1		 ?		 ?

C INPUT PARAMETERS
define(`rp', `r3')
define(`up', `r4')
define(`n',  `r5')
define(`vp', `r6')

define(`cy0', `r10')
ifdef(`EXTRA_REGISTER',
` define(`cy1', EXTRA_REGISTER)',
` define(`cy1', `r31')')

ifdef(`OPERATION_mul_2',`
  define(`AM',		`')
  define(`ADDX',	`addc')
  define(`func',	`mpn_mul_2')
')
ifdef(`OPERATION_addmul_2',`
  define(`AM',		`$1')
  define(`ADDX',	`adde')
  define(`func',	`mpn_addmul_2')
')

MULFUNC_PROLOGUE(mpn_mul_2 mpn_addmul_2)

ASM_START()
PROLOGUE(func)

ifdef(`EXTRA_REGISTER',,`
	std	r31, -8(r1)
')
	andi.	r12, n, 1
	addi	r0, n, 1
	srdi	r0, r0, 1
	mtctr	r0
	ld	r11, 0(vp)		C v0
	li	cy0, 0
	ld	r12, 8(vp)		C v1
	li	cy1, 0
	ld	r5, 0(up)
	beq	L(lo0)
	addi	up, up, -8
	addi	rp, rp, -8
	b	L(lo1)

	ALIGN(32)
L(top):
AM(`	ld	r0, -8(rp)')
	ld	r5, 0(up)
AM(`	addc	r6, r6, r0')
	ADDX	r7, r7, r8
	addze	r9, r9
	addc	r6, r6, cy0
	adde	cy0, r7, cy1
	std	r6, -8(rp)
	addze	cy1, r9
L(lo0):	mulld	r6, r11, r5		C v0 * u[i]  weight 0
	mulhdu	r7, r11, r5		C v0 * u[i]  weight 1
	mulld	r8, r12, r5		C v1 * u[i]  weight 1
	mulhdu	r9, r12, r5		C v1 * u[i]  weight 2
AM(`	ld	r0, 0(rp)')
	ld	r5, 8(up)
AM(`	addc	r6, r6, r0')
	ADDX	r7, r7, r8
	addze	r9, r9
	addc	r6, r6, cy0
	adde	cy0, r7, cy1
	std	r6, 0(rp)
	addze	cy1, r9
L(lo1):	mulld	r6, r11, r5		C v0 * u[i]  weight 0
	mulhdu	r7, r11, r5		C v0 * u[i]  weight 1
	addi	up, up, 16
	addi	rp, rp, 16
	mulld	r8, r12, r5		C v1 * u[i]  weight 1
	mulhdu	r9, r12, r5		C v1 * u[i]  weight 2
	bdnz	L(top)

L(end):
AM(`	ld	r0, -8(rp)')
AM(`	addc	r6, r6, r0')
	ADDX	r7, r7, r8
	addze	r9, r9
	addc	r6, r6, cy0
	std	r6, -8(rp)
	adde	cy0, r7, cy1
	addze	cy1, r9
	std	cy0, 0(rp)
	mr	r3, cy1

ifdef(`EXTRA_REGISTER',,`
	ld	r31, -8(r1)
')
	blr
EPILOGUE()
