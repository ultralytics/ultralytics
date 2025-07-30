dnl  ARM mpn_add_n and mpn_sub_n

dnl  Contributed to the GNU project by Robert Harley.

dnl  Copyright 1997, 2000, 2001, 2012 Free Software Foundation, Inc.

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

C	     cycles/limb
C StrongARM	 ?
C XScale	 ?
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	 2.5	slightly fluctuating
C Cortex-A15	 2.25

define(`rp', `r0')
define(`up', `r1')
define(`vp', `r2')
define(`n',  `r3')

ifdef(`OPERATION_add_n', `
  define(`ADDSUB',	adds)
  define(`ADDSUBC',	adcs)
  define(`CLRCY',	`cmn	r0, #0')
  define(`SETCY',	`cmp	$1, #1')
  define(`RETVAL',	`adc	r0, n, #0')
  define(`func',	mpn_add_n)
  define(`func_nc',	mpn_add_nc)')
ifdef(`OPERATION_sub_n', `
  define(`ADDSUB',	subs)
  define(`ADDSUBC',	sbcs)
  define(`CLRCY',	`cmp	r0, r0')
  define(`SETCY',	`rsbs	$1, $1, #0')
  define(`RETVAL',	`sbc	r0, r0, r0
			and	r0, r0, #1')
  define(`func',	mpn_sub_n)
  define(`func_nc',	mpn_sub_nc)')

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc mpn_sub_n mpn_sub_nc)

ASM_START()
PROLOGUE(func_nc)
	ldr	r12, [sp, #0]
	stmfd	sp!, { r8, r9, lr }
	SETCY(	r12)
	b	L(ent)
EPILOGUE()
PROLOGUE(func)
	stmfd	sp!, { r8, r9, lr }
	CLRCY(	r12)
L(ent):	tst	n, #1
	beq	L(skip1)
	ldr	r12, [up], #4
	ldr	lr, [vp], #4
	ADDSUBC	r12, r12, lr
	str	r12, [rp], #4
L(skip1):
	tst	n, #2
	beq	L(skip2)
	ldmia	up!, { r8, r9 }
	ldmia	vp!, { r12, lr }
	ADDSUBC	r8, r8, r12
	ADDSUBC	r9, r9, lr
	stmia	rp!, { r8, r9 }
L(skip2):
	bics	n, n, #3
	beq	L(rtn)
	stmfd	sp!, { r4, r5, r6, r7 }

L(top):	ldmia	up!, { r4, r5, r6, r7 }
	ldmia	vp!, { r8, r9, r12, lr }
	ADDSUBC	r4, r4, r8
	sub	n, n, #4
	ADDSUBC	r5, r5, r9
	ADDSUBC	r6, r6, r12
	ADDSUBC	r7, r7, lr
	stmia	rp!, { r4, r5, r6, r7 }
	teq	n, #0
	bne	L(top)

	ldmfd	sp!, { r4, r5, r6, r7 }

L(rtn):	RETVAL
	ldmfd	sp!, { r8, r9, pc }
EPILOGUE()
