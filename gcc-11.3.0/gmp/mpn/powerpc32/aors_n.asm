dnl  PowerPC-32 mpn_add_n and mpn_sub_n.

dnl  Copyright 2002, 2005, 2007 Free Software Foundation, Inc.

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
C 603e:                  ?
C 604e:                  ?		old: 3.25
C 75x (G3):              ?		old: 3.5
C 7400,7410 (G4):        3.25
C 744x,745x (G4+):       4
C POWER3/PPC630          2
C POWER4/PPC970          2.4
C POWER5                 2.75
C POWER6               40-140
C POWER7                 3

C INPUT PARAMETERS
define(`rp',	`r3')
define(`up',	`r4')
define(`vp',	`r5')
define(`n',	`r6')
define(`cy',	`r7')

ifdef(`OPERATION_add_n', `
	define(ADCSBC,	adde)
	define(func,	mpn_add_n)
	define(func_nc,	mpn_add_nc)
	define(IFADD,	`$1')
	define(IFSUB,	`')')
ifdef(`OPERATION_sub_n', `
	define(ADCSBC,	subfe)
	define(func,	mpn_sub_n)
	define(func_nc,	mpn_sub_nc)
	define(IFADD,	`')
	define(IFSUB,	`$1')')

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc mpn_sub_n mpn_sub_nc)

ASM_START()

PROLOGUE(func_nc)
IFADD(`	addic	r0, cy, -1')		C set carry from argument
IFSUB(`	subfic	r0, cy, 0')		C set carry from argument
	b	L(ent)
EPILOGUE()

PROLOGUE(func)
IFADD(`	addic	r0, n, 0')		C clear carry
IFSUB(`	addic	r0, n, -1')		C set carry
L(ent):	andi.	r0, n, 3
	addi	r3, r3, -12
	addi	n, n, 1
	cmpwi	cr7, r0, 2
	srwi	r0, n, 2
	sub	r4, r4, r3
	sub	r5, r5, r3
	mtctr	r0
	bne	cr0, L(n00)

	lwzx	r7, r4, r3		C n = 4, 8, 12, ...
	lwzx	r8, r5, r3
	addi	r3, r3, 4
	lwzx	r9, r4, r3
	ADCSBC	r7, r8, r7
	lwzx	r10, r5, r3
	addi	r3, r3, 4
	b	L(00)

L(n00):	bge	cr7, L(n01)
	cmpwi	cr0, r0, 0		C n = 1, 5, 9, 13, ...
	lwzx	r0, r4, r3
	lwzx	r6, r5, r3
	addi	r3, r3, 4
	ADCSBC	r0, r6, r0
	ble	L(ret)
L(gt1):	lwzx	r7, r4, r3
	lwzx	r8, r5, r3
	addi	r3, r3, 4
	b	L(01)

L(n10):
	lwzx	r9, r4, r3		C n = 3, 7, 11, 15, ...
	lwzx	r10, r5, r3
	addi	r3, r3, 4
	lwzx	r11, r4, r3
	ADCSBC	r9, r10, r9
	lwzx	r12, r5, r3
	addi	r3, r3, 4
	b	L(11)

L(n01):	bne	cr7, L(n10)
	cmpwi	cr0, r0, 0		C n = 2, 6, 10, 14, ...
	lwzx	r11, r4, r3
	lwzx	r12, r5, r3
	addi	r3, r3, 4
	lwzx	r0, r4, r3
	ADCSBC	r11, r12, r11
	lwzx	r6, r5, r3
	addi	r3, r3, 4
	ble	cr0, L(end)


L(lp):	lwzx	r7, r4, r3
	ADCSBC	r0, r6, r0
	lwzx	r8, r5, r3
	stwu	r11, 4(r3)
L(01):	lwzx	r9, r4, r3
	ADCSBC	r7, r8, r7
	lwzx	r10, r5, r3
	stwu	r0, 4(r3)
L(00):	lwzx	r11, r4, r3
	ADCSBC	r9, r10, r9
	lwzx	r12, r5, r3
	stwu	r7, 4(r3)
L(11):	lwzx	r0, r4, r3
	ADCSBC	r11, r12, r11
	lwzx	r6, r5, r3
	stwu	r9, 4(r3)
	bdnz	L(lp)

L(end):	ADCSBC	r0, r6, r0
	stw	r11, 4(r3)
L(ret):	stw	r0, 8(r3)
IFADD(`	li	r3, 0	')
IFADD(`	addze	r3, r3	')
IFSUB(`	subfe	r3, r0, r0')
IFSUB(`	neg	r3, r3')
	blr
EPILOGUE()
