dnl  ARM mpn_and_n, mpn_andn_n. mpn_nand_n, etc, optimised for A15.

dnl  Contributed to the GNU project by Torbjörn Granlund.

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

C            cycles/limb             cycles/limb
C          and andn ior xor         nand iorn nior xnor
C StrongARM	 ?			 ?
C XScale	 ?			 ?
C Cortex-A7	 ?			 ?
C Cortex-A8	 ?			 ?
C Cortex-A9	3.5			3.56
C Cortex-A15	1.27			1.64

C This is great A15 core register code, but it is a bit large.
C We use FEEDIN_VARIANT 1 to save some space, but use 8-way unrolling.

C Architecture requirements:
C v5	-
C v5t	-
C v5te	ldrd strd
C v6	-
C v6t2	-
C v7a	-

define(`FEEDIN_VARIANT', 1)	C alternatives: 0 1 2
define(`UNROLL', 4x2)		C alternatives: 4 4x2

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
	push	{ r4-r9 }

ifelse(FEEDIN_VARIANT,0,`
	ands	r6, n, #3
	mov	n, n, lsr #2
	beq	L(b00a)
	tst	r6, #1
	beq	L(bx0)
	ldr	r5, [up], #4
	ldr	r7, [vp], #4
	LOGOP(	r9, r5, r7)
	POSTOP(	r9)
	str	r9, [rp], #4
	tst	r6, #2
	beq	L(b00)
L(bx0):	ldrd	r4, r5, [up, #0]
	ldrd	r6, r7, [vp, #0]
	sub	rp, rp, #8
	b	L(lo)
L(b00):	tst	n, n
	beq	L(wd1)
L(b00a):ldrd	r4, r5, [up], #-8
	ldrd	r6, r7, [vp], #-8
	sub	rp, rp, #16
	b	L(mid)
')
ifelse(FEEDIN_VARIANT,1,`
	and	r6, n, #3
	mov	n, n, lsr #2
	tst	r6, #1
	beq	L(bx0)
	ldr	r5, [up], #4
	ldr	r7, [vp], #4
	LOGOP(	r9, r5, r7)
	POSTOP(	r9)
	str	r9, [rp], #4
L(bx0):	tst	r6, #2
	beq	L(b00)
	ldrd	r4, r5, [up, #0]
	ldrd	r6, r7, [vp, #0]
	sub	rp, rp, #8
	b	L(lo)
L(b00):	tst	n, n
	beq	L(wd1)
	ldrd	r4, r5, [up], #-8
	ldrd	r6, r7, [vp], #-8
	sub	rp, rp, #16
	b	L(mid)
')
ifelse(FEEDIN_VARIANT,2,`
	ands	r6, n, #3
	mov	n, n, lsr #2
	beq	L(b00)
	cmp	r6, #2
	bcc	L(b01)
	beq	L(b10)

L(b11):	ldr	r5, [up], #4
	ldr	r7, [vp], #4
	LOGOP(	r9, r5, r7)
	ldrd	r4, r5, [up, #0]
	ldrd	r6, r7, [vp, #0]
	POSTOP(	r9)
	str	r9, [rp], #-4
	b	L(lo)

L(b00):	ldrd	r4, r5, [up], #-8
	ldrd	r6, r7, [vp], #-8
	sub	rp, rp, #16
	b	L(mid)

L(b01):	ldr	r5, [up], #-4
	ldr	r7, [vp], #-4
	LOGOP(	r9, r5, r7)
	POSTOP(	r9)
	str	r9, [rp], #-12
	tst	n, n
	beq	L(wd1)
L(gt1):	ldrd	r4, r5, [up, #8]
	ldrd	r6, r7, [vp, #8]
	b	L(mid)

L(b10):	ldrd	r4, r5, [up]
	ldrd	r6, r7, [vp]
	sub	rp, rp, #8
	b	L(lo)
')
	ALIGN(16)
ifelse(UNROLL,4,`
L(top):	ldrd	r4, r5, [up, #8]
	ldrd	r6, r7, [vp, #8]
	POSTOP(	r8)
	POSTOP(	r9)
	strd	r8, r9, [rp, #8]
L(mid):	LOGOP(	r8, r4, r6)
	LOGOP(	r9, r5, r7)
	ldrd	r4, r5, [up, #16]!
	ldrd	r6, r7, [vp, #16]!
	POSTOP(	r8)
	POSTOP(	r9)
	strd	r8, r9, [rp, #16]!
	sub	n, n, #1
L(lo):	LOGOP(	r8, r4, r6)
	LOGOP(	r9, r5, r7)
	tst	n, n
	bne	L(top)
')
ifelse(UNROLL,4x2,`
L(top):	ldrd	r4, r5, [up, #8]
	ldrd	r6, r7, [vp, #8]
	POSTOP(	r8)
	POSTOP(	r9)
	strd	r8, r9, [rp, #8]
L(mid):	LOGOP(	r8, r4, r6)
	LOGOP(	r9, r5, r7)
	ldrd	r4, r5, [up, #16]
	ldrd	r6, r7, [vp, #16]
	POSTOP(	r8)
	POSTOP(	r9)
	strd	r8, r9, [rp, #16]
	LOGOP(	r8, r4, r6)
	LOGOP(	r9, r5, r7)
	sub	n, n, #2
	tst	n, n
	bmi	L(dne)
	ldrd	r4, r5, [up, #24]
	ldrd	r6, r7, [vp, #24]
	POSTOP(	r8)
	POSTOP(	r9)
	strd	r8, r9, [rp, #24]
	LOGOP(	r8, r4, r6)
	LOGOP(	r9, r5, r7)
	ldrd	r4, r5, [up, #32]!
	ldrd	r6, r7, [vp, #32]!
	POSTOP(	r8)
	POSTOP(	r9)
	strd	r8, r9, [rp, #32]!
L(lo):	LOGOP(	r8, r4, r6)
	LOGOP(	r9, r5, r7)
	tst	n, n
	bne	L(top)
')

L(end):	POSTOP(	r8)
	POSTOP(	r9)
	strd	r8, r9, [rp, #8]
L(wd1):	pop	{ r4-r9 }
	bx	r14
ifelse(UNROLL,4x2,`
L(dne):	POSTOP(	r8)
	POSTOP(	r9)
	strd	r8, r9, [rp, #24]
	pop	{ r4-r9 }
	bx	r14
')
EPILOGUE()
