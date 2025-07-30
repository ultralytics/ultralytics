dnl  ARM mpn_addlsh1_n and mpn_sublsh1_n

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2012 Free Software Foundation, Inc.

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

C	      addlsh1_n       sublsh1_n
C	     cycles/limb     cycles/limb
C StrongARM	 ?		 ?
C XScale	 ?		 ?
C Cortex-A7	 ?		 ?
C Cortex-A8	 ?		 ?
C Cortex-A9	 3.12		 3.7
C Cortex-A15	 ?		 ?

C TODO
C  * The addlsh1_n code runs well, but is only barely faster than mpn_addmul_1.
C    The sublsh1_n code could surely be tweaked, its REVCY slows down things
C    very much.  If two insns are really needed, it might help to separate them
C    for better micro-parallelism.

define(`rp', `r0')
define(`up', `r1')
define(`vp', `r2')
define(`n',  `r3')

ifdef(`OPERATION_addlsh1_n', `
  define(`ADDSUB',	adds)
  define(`ADDSUBC',	adcs)
  define(`SETCY',	`cmp	$1, #1')
  define(`RETVAL',	`adc	r0, $1, #2')
  define(`SAVECY',	`sbc	$1, $2, #0')
  define(`RESTCY',	`cmn	$1, #1')
  define(`REVCY',	`')
  define(`INICYR',	`mov	$1, #0')
  define(`r10r11',	`r11')
  define(`func',	mpn_addlsh1_n)
  define(`func_nc',	mpn_addlsh1_nc)')
ifdef(`OPERATION_sublsh1_n', `
  define(`ADDSUB',	subs)
  define(`ADDSUBC',	sbcs)
  define(`SETCY',	`rsbs	$1, $1, #0')
  define(`RETVAL',	`adc	r0, $1, #1')
  define(`SAVECY',	`sbc	$1, $1, $1')
  define(`RESTCY',	`cmn	$1, #1')
  define(`REVCY',	`sbc	$1, $1, $1
			cmn	$1, #1')
  define(`INICYR',	`mvn	$1, #0')
  define(`r10r11',	`r10')
  define(`func',	mpn_sublsh1_n)
  define(`func_nc',	mpn_sublsh1_nc)')

MULFUNC_PROLOGUE(mpn_addlsh1_n mpn_sublsh1_n)

ASM_START()
PROLOGUE(func)
	push	{r4-r10r11, r14}

ifdef(`OPERATION_addlsh1_n', `
	mvn	r11, #0
')
	INICYR(	r14)
	subs	n, n, #3
	blt	L(le2)			C carry clear on branch path

	cmn	r0, #0			C clear carry
	ldmia	vp!, {r8, r9, r10}
	b	L(mid)

L(top):	RESTCY(	r14)
	ADDSUBC	r4, r4, r8
	ADDSUBC	r5, r5, r9
	ADDSUBC	r6, r6, r10
	ldmia	vp!, {r8, r9, r10}
	stmia	rp!, {r4, r5, r6}
	REVCY(r14)
	adcs	r8, r8, r8
	adcs	r9, r9, r9
	adcs	r10, r10, r10
	ldmia	up!, {r4, r5, r6}
	SAVECY(	r14, r11)
	subs	n, n, #3
	blt	L(exi)
	RESTCY(	r12)
	ADDSUBC	r4, r4, r8
	ADDSUBC	r5, r5, r9
	ADDSUBC	r6, r6, r10
	ldmia	vp!, {r8, r9, r10}
	stmia	rp!, {r4, r5, r6}
	REVCY(r12)
L(mid):	adcs	r8, r8, r8
	adcs	r9, r9, r9
	adcs	r10, r10, r10
	ldmia	up!, {r4, r5, r6}
	SAVECY(	r12, r11)
	subs	n, n, #3
	bge	L(top)

	mov	r7, r12			C swap alternating...
	mov	r12, r14		C ...carry-save...
	mov	r14, r7			C ...registers

L(exi):	RESTCY(	r12)
	ADDSUBC	r4, r4, r8
	ADDSUBC	r5, r5, r9
	ADDSUBC	r6, r6, r10
	stmia	rp!, {r4, r5, r6}

	REVCY(r12)
L(le2):	tst	n, #1			C n = {-1,-2,-3} map to [2], [1], [0]
	beq	L(e1)

L(e02):	tst	n, #2
	beq	L(rt0)
	ldm	vp, {r8, r9}
	adcs	r8, r8, r8
	adcs	r9, r9, r9
	ldm	up, {r4, r5}
	SAVECY(	r12, r11)
	RESTCY(	r14)
	ADDSUBC	r4, r4, r8
	ADDSUBC	r5, r5, r9
	stm	rp, {r4, r5}
	b	L(rt1)

L(e1):	ldr	r8, [vp]
	adcs	r8, r8, r8
	ldr	r4, [up]
	SAVECY(	r12, r11)
	RESTCY(	r14)
	ADDSUBC	r4, r4, r8
	str	r4, [rp]

L(rt1):	mov	r14, r12
	REVCY(r12)
L(rt0):	RETVAL(	r14)
	pop	{r4-r10r11, r14}
	bx	r14
EPILOGUE()
