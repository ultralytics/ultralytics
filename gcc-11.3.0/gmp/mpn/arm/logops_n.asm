dnl  ARM mpn_and_n, mpn_andn_n. mpn_nand_n, etc.

dnl  Contributed to the GNU project by Torbjörn Granlund.

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

C            cycles/limb             cycles/limb
C          and andn ior xor         nand iorn nior xnor
C StrongARM	 ?			 ?
C XScale	 ?			 ?
C Cortex-A7	 ?			 ?
C Cortex-A8	 ?			 ?
C Cortex-A9	2.5-2.72		2.75-3
C Cortex-A15	2.25			2.75

C TODO
C  * It seems that 2.25 c/l and 2.75 c/l is possible for A9.
C  * Debug popping issue, see comment below.

define(`rp', `r0')
define(`up', `r1')
define(`vp', `r2')
define(`n',  `r3')

define(`POSTOP')

ifdef(`OPERATION_and_n',`
  define(`func',    `mpn_and_n')
  define(`LOGOP',   `and	$1, $2, $3')')
ifdef(`OPERATION_andn_n',`
  define(`func',    `mpn_andn_n')
  define(`LOGOP',   `bic	$1, $2, $3')')
ifdef(`OPERATION_nand_n',`
  define(`func',    `mpn_nand_n')
  define(`POSTOP',  `mvn	$1, $1')
  define(`LOGOP',   `and	$1, $2, $3')')
ifdef(`OPERATION_ior_n',`
  define(`func',    `mpn_ior_n')
  define(`LOGOP',   `orr	$1, $2, $3')')
ifdef(`OPERATION_iorn_n',`
  define(`func',    `mpn_iorn_n')
  define(`POSTOP',  `mvn	$1, $1')
  define(`LOGOP',   `bic	$1, $3, $2')')
ifdef(`OPERATION_nior_n',`
  define(`func',    `mpn_nior_n')
  define(`POSTOP',  `mvn	$1, $1')
  define(`LOGOP',   `orr	$1, $2, $3')')
ifdef(`OPERATION_xor_n',`
  define(`func',    `mpn_xor_n')
  define(`LOGOP',   `eor	$1, $2, $3')')
ifdef(`OPERATION_xnor_n',`
  define(`func',    `mpn_xnor_n')
  define(`POSTOP',  `mvn	$1, $1')
  define(`LOGOP',   `eor	$1, $2, $3')')

MULFUNC_PROLOGUE(mpn_and_n mpn_andn_n mpn_nand_n mpn_ior_n mpn_iorn_n mpn_nior_n mpn_xor_n mpn_xnor_n)

ASM_START()
PROLOGUE(func)
	push	{ r8, r9, r10 }
	tst	n, #1
	beq	L(skip1)
	ldr	r10, [vp], #4
	ldr	r12, [up], #4
	LOGOP(	r12, r12, r10)
	POSTOP(	r12)
	str	r12, [rp], #4
L(skip1):
	tst	n, #2
	beq	L(skip2)
	ldmia	vp!, { r10, r12 }
	ldmia	up!, { r8, r9 }
	LOGOP(	r8, r8, r10)
	LOGOP(	r9, r9, r12)
	POSTOP(	r8)
	POSTOP(	r9)
	stmia	rp!, { r8, r9 }
L(skip2):
	bics	n, n, #3
	beq	L(rtn)
	push	{ r4, r5, r6, r7 }

	ldmia	vp!, { r8, r9, r10, r12 }
	b	L(mid)

L(top):	ldmia	vp!, { r8, r9, r10, r12 }
	POSTOP(	r4)
	POSTOP(	r5)
	POSTOP(	r6)
	POSTOP(	r7)
	stmia	rp!, { r4, r5, r6, r7 }
L(mid):	sub	n, n, #4
	ldmia	up!, { r4, r5, r6, r7 }
	teq	n, #0
	LOGOP(	r4, r4, r8)
	LOGOP(	r5, r5, r9)
	LOGOP(	r6, r6, r10)
	LOGOP(	r7, r7, r12)
	bne	L(top)

	POSTOP(	r4)
	POSTOP(	r5)
	POSTOP(	r6)
	POSTOP(	r7)
	stmia	rp!, { r4, r5, r6, r7 }

	pop	{ r4, r5, r6, r7 }	C popping r8-r10 here strangely fails

L(rtn):	pop	{ r8, r9, r10 }
	bx	r14
EPILOGUE()
