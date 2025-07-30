dnl  ARM mpn_rsh1add_n and mpn_rsh1sub_n.

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

C	     cycles/limb
C StrongARM	 ?
C XScale	 ?
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	3.64-3.7
C Cortex-A15	 2.5

C TODO
C  * Not optimised.

define(`rp', `r0')
define(`up', `r1')
define(`vp', `r2')
define(`n',  `r3')

ifdef(`OPERATION_rsh1add_n', `
  define(`ADDSUB',	adds)
  define(`ADDSUBC',	adcs)
  define(`RSTCY',	`cmn	$1, $1')
  define(`func',	mpn_rsh1add_n)
  define(`func_nc',	mpn_rsh1add_nc)')
ifdef(`OPERATION_rsh1sub_n', `
  define(`ADDSUB',	subs)
  define(`ADDSUBC',	sbcs)
  define(`RSTCY',
	`mvn	$2, #0x80000000
	cmp	$2, $1')
  define(`func',	mpn_rsh1sub_n)
  define(`func_nc',	mpn_rsh1sub_nc)')

MULFUNC_PROLOGUE(mpn_rsh1add_n mpn_rsh1sub_n)

ASM_START()
PROLOGUE(func)
	push	{r4-r11}
	ldr	r4, [up], #4
	ldr	r8, [vp], #4
	ADDSUB	r4, r4, r8
	movs	r12, r7, rrx
	and	r11, r4, #1	C return value
	subs	n, n, #4
	blo	L(end)

L(top):	ldmia	up!, {r5,r6,r7}
	ldmia	vp!, {r8,r9,r10}
	cmn	r12, r12
	ADDSUBC	r5, r5, r8
	ADDSUBC	r6, r6, r9
	ADDSUBC	r7, r7, r10
	movs	r12, r7, rrx
	movs	r6, r6, rrx
	movs	r5, r5, rrx
	movs	r4, r4, rrx
	subs	n, n, #3
	stmia	rp!, {r4,r5,r6}
	mov	r4, r7
	bhs	L(top)

L(end):	cmn	n, #2
	bls	L(e2)
	ldm	up, {r5,r6}
	ldm	vp, {r8,r9}
	cmn	r12, r12
	ADDSUBC	r5, r5, r8
	ADDSUBC	r6, r6, r9
	movs	r12, r6, rrx
	movs	r5, r5, rrx
	movs	r4, r4, rrx
	stmia	rp!, {r4,r5}
	mov	r4, r6
	b	L(e1)

L(e2):	bne	L(e1)
	ldr	r5, [up, #0]
	ldr	r8, [vp, #0]
	cmn	r12, r12
	ADDSUBC	r5, r5, r8
	movs	r12, r5, rrx
	movs	r4, r4, rrx
	str	r4, [rp], #4
	mov	r4, r5

L(e1):	RSTCY(	r12, r1)
	mov	r4, r4, rrx
	str	r4, [rp, #0]
	mov	r0, r11
	pop	{r4-r11}
	bx	r14
EPILOGUE()
