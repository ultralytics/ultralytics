dnl  ARM mpn_addmul_1 and mpn_submul_1.

dnl  Copyright 1998, 2000, 2001, 2003, 2012 Free Software Foundation, Inc.

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
C StrongARM:     ?
C XScale	 ?
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	 5.25
C Cortex-A15	 4

define(`rp', `r0')
define(`up', `r1')
define(`n',  `r2')
define(`vl', `r3')
define(`rl', `r12')
define(`ul', `r6')
define(`r',  `lr')

ifdef(`OPERATION_addmul_1', `
  define(`ADDSUB',	adds)
  define(`ADDSUBC',	adcs)
  define(`CLRRCY',	`mov	$1, #0
			adds	r0, r0, #0')
  define(`RETVAL',	`adc	r0, r4, #0')
  define(`func',	mpn_addmul_1)')
ifdef(`OPERATION_submul_1', `
  define(`ADDSUB',	subs)
  define(`ADDSUBC',	sbcs)
  define(`CLRRCY',	`subs	$1, r0, r0')
  define(`RETVAL',	`sbc	r0, r0, r0
			sub	r0, $1, r0')
  define(`func',	mpn_submul_1)')

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_submul_1)

ASM_START()
PROLOGUE(func)
	stmfd	sp!, { r4-r6, lr }
	CLRRCY(	r4)
	tst	n, #1
	beq	L(skip1)
	ldr	ul, [up], #4
	ldr	rl, [rp, #0]
	umull	r5, r4, ul, vl
	ADDSUB	r, rl, r5
	str	r, [rp], #4
L(skip1):
	tst	n, #2
	beq	L(skip2)
	ldr	ul, [up], #4
	ldr	rl, [rp, #0]
	mov	r5, #0
	umlal	r4, r5, ul, vl
	ldr	ul, [up], #4
	ADDSUBC	r, rl, r4
	ldr	rl, [rp, #4]
	mov	r4, #0
	umlal	r5, r4, ul, vl
	str	r, [rp], #4
	ADDSUBC	r, rl, r5
	str	r, [rp], #4
L(skip2):
	bics	n, n, #3
	beq	L(rtn)

	ldr	ul, [up], #4
	ldr	rl, [rp, #0]
	mov	r5, #0
	umlal	r4, r5, ul, vl
	b	L(in)

L(top):	ldr	ul, [up], #4
	ADDSUBC	r, rl, r5
	ldr	rl, [rp, #4]
	mov	r5, #0
	umlal	r4, r5, ul, vl
	str	r, [rp], #4
L(in):	ldr	ul, [up], #4
	ADDSUBC	r, rl, r4
	ldr	rl, [rp, #4]
	mov	r4, #0
	umlal	r5, r4, ul, vl
	str	r, [rp], #4
	ldr	ul, [up], #4
	ADDSUBC	r, rl, r5
	ldr	rl, [rp, #4]
	mov	r5, #0
	umlal	r4, r5, ul, vl
	str	r, [rp], #4
	ldr	ul, [up], #4
	ADDSUBC	r, rl, r4
	ldr	rl, [rp, #4]
	mov	r4, #0
	umlal	r5, r4, ul, vl
	sub	n, n, #4
	tst	n, n
	str	r, [rp], #4
	bne	L(top)

	ADDSUBC	r, rl, r5
	str	r, [rp]

L(rtn):	RETVAL(	r4)
	ldmfd	sp!, { r4-r6, pc }
EPILOGUE()
