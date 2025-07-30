dnl  ARM mpn_cnd_add_n/mpn_cnd_sub_n optimised for A15.

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

C	     cycles/limb		best
C StrongARM:     -
C XScale	 ?
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	 3.75			 3
C Cortex-A15	 1.78			this

C This code does not run as well as one could have hoped, since 1.5 c/l seems
C realistic for this insn mix.

C Architecture requirements:
C v5	-
C v5t	-
C v5te	ldrd strd
C v6	-
C v6t2	-
C v7a	-

define(`cnd',`r0')
define(`rp', `r1')
define(`up', `r2')
define(`vp', `r3')
define(`n',  `r12')

ifdef(`OPERATION_cnd_add_n', `
  define(`ADDSUB',	adds)
  define(`ADDSUBC',	adcs)
  define(`IFADD',	`$1')
  define(`INITCY',      `cmn	r0, #0')
  define(`RETVAL',	`adc	r0, n, #0')
  define(`RETVAL2',	`adc	r0, n, #1')
  define(`func',	mpn_cnd_add_n)
  define(`func_nc',	mpn_add_nc)')
ifdef(`OPERATION_cnd_sub_n', `
  define(`ADDSUB',	subs)
  define(`ADDSUBC',	sbcs)
  define(`IFADD',	`')
  define(`INITCY',      `cmp	r0, #0')
  define(`RETVAL',	`sbc	r0, r0, r0
			and	r0, r0, #1')
  define(`RETVAL2',	`RETVAL')
  define(`func',	mpn_cnd_sub_n)
  define(`func_nc',	mpn_sub_nc)')

MULFUNC_PROLOGUE(mpn_cnd_add_n mpn_cnd_sub_n)

ASM_START()
PROLOGUE(func)
	ldr	n, [sp]
	push	{ r4-r9 }

	cmp	cnd, #1
	sbc	cnd, cnd, cnd		C conditionally set to 0xffffffff

	ands	r6, n, #3
	mov	n, n, lsr #2
	beq	L(b00)
	cmp	r6, #2
	bcc	L(b01)
	beq	L(b10)

L(b11):	ldr	r5, [up], #4
	ldr	r7, [vp], #4
	bic	r7, r7, cnd
	ADDSUB	r9, r5, r7
	ldrd	r4, r5, [up, #0]
	ldrd	r6, r7, [vp, #0]
	bic	r6, r6, cnd
	bic	r7, r7, cnd
	str	r9, [rp], #-4
	b	L(lo)

L(b00):	ldrd	r4, r5, [up], #-8
	ldrd	r6, r7, [vp], #-8
	bic	r6, r6, cnd
	bic	r7, r7, cnd
	INITCY
	sub	rp, rp, #16
	b	L(mid)

L(b01):	ldr	r5, [up], #-4
	ldr	r7, [vp], #-4
	bic	r7, r7, cnd
	ADDSUB	r9, r5, r7
	str	r9, [rp], #-12
	tst	n, n
	beq	L(wd1)
L(gt1):	ldrd	r4, r5, [up, #8]
	ldrd	r6, r7, [vp, #8]
	bic	r6, r6, cnd
	bic	r7, r7, cnd
	b	L(mid)

L(b10):	ldrd	r4, r5, [up]
	ldrd	r6, r7, [vp]
	bic	r6, r6, cnd
	bic	r7, r7, cnd
	INITCY
	sub	rp, rp, #8
	b	L(lo)

	ALIGN(16)
L(top):	ldrd	r6, r7, [vp, #8]
	ldrd	r4, r5, [up, #8]
	bic	r6, r6, cnd
	bic	r7, r7, cnd
	strd	r8, r9, [rp, #8]
L(mid):	ADDSUBC	r8, r4, r6
	ADDSUBC	r9, r5, r7
	ldrd	r6, r7, [vp, #16]!
	ldrd	r4, r5, [up, #16]!
	bic	r6, r6, cnd
	bic	r7, r7, cnd
	sub	n, n, #1
	strd	r8, r9, [rp, #16]!
L(lo):	ADDSUBC	r8, r4, r6
	ADDSUBC	r9, r5, r7
	tst	n, n
	bne	L(top)

L(end):	strd	r8, r9, [rp, #8]
L(wd1):	RETVAL
	pop	{ r4-r9 }
	bx	r14
EPILOGUE()
