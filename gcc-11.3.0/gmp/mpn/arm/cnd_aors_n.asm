dnl  ARM mpn_cnd_add_n, mpn_cnd_sub_n

dnl  Copyright 2012, 2013 Free Software Foundation, Inc.

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
C Cortex-A9	 3
C Cortex-A15	 2.5

define(`cnd',	`r0')
define(`rp',	`r1')
define(`up',	`r2')
define(`vp',	`r3')

define(`n',	`r12')


ifdef(`OPERATION_cnd_add_n', `
	define(`ADDSUB',      adds)
	define(`ADDSUBC',      adcs)
	define(`INITCY',      `cmn	r0, #0')
	define(`RETVAL',      `adc	r0, n, #0')
	define(func,	      mpn_cnd_add_n)')
ifdef(`OPERATION_cnd_sub_n', `
	define(`ADDSUB',      subs)
	define(`ADDSUBC',      sbcs)
	define(`INITCY',      `cmp	r0, #0')
	define(`RETVAL',      `adc	r0, n, #0
			      rsb	r0, r0, #1')
	define(func,	      mpn_cnd_sub_n)')

MULFUNC_PROLOGUE(mpn_cnd_add_n mpn_cnd_sub_n)

ASM_START()
PROLOGUE(func)
	push	{r4-r11}
	ldr	n, [sp, #32]

	cmp	cnd, #1
	sbc	cnd, cnd, cnd		C conditionally set to 0xffffffff

	INITCY				C really only needed for n = 0 (mod 4)

	ands	r4, n, #3
	beq	L(top)
	cmp	r4, #2
	bcc	L(b1)
	beq	L(b2)

L(b3):	ldm	vp!, {r4,r5,r6}
	ldm	up!, {r8,r9,r10}
	bic	r4, r4, cnd
	bic	r5, r5, cnd
	bic	r6, r6, cnd
	ADDSUB	r8, r8, r4
	ADDSUBC	r9, r9, r5
	ADDSUBC	r10, r10, r6
	stm	rp!, {r8,r9,r10}
	sub	n, n, #3
	teq	n, #0
	bne	L(top)
	b	L(end)

L(b2):	ldm	vp!, {r4,r5}
	ldm	up!, {r8,r9}
	bic	r4, r4, cnd
	bic	r5, r5, cnd
	ADDSUB	r8, r8, r4
	ADDSUBC	r9, r9, r5
	stm	rp!, {r8,r9}
	sub	n, n, #2
	teq	n, #0
	bne	L(top)
	b	L(end)

L(b1):	ldr	r4, [vp], #4
	ldr	r8, [up], #4
	bic	r4, r4, cnd
	ADDSUB	r8, r8, r4
	str	r8, [rp], #4
	sub	n, n, #1
	teq	n, #0
	beq	L(end)

L(top):	ldm	vp!, {r4,r5,r6,r7}
	ldm	up!, {r8,r9,r10,r11}
	bic	r4, r4, cnd
	bic	r5, r5, cnd
	bic	r6, r6, cnd
	bic	r7, r7, cnd
	ADDSUBC	r8, r8, r4
	ADDSUBC	r9, r9, r5
	ADDSUBC	r10, r10, r6
	ADDSUBC	r11, r11, r7
	sub	n, n, #4
	stm	rp!, {r8,r9,r10,r11}
	teq	n, #0
	bne	L(top)

L(end):	RETVAL
	pop	{r4-r11}
	bx	r14
EPILOGUE()
