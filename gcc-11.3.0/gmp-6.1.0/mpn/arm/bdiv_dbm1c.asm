dnl  ARM mpn_bdiv_dbm1c.

dnl  Copyright 2008, 2011, 2012 Free Software Foundation, Inc.

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
C Cortex-A9	 4.25
C Cortex-A15	 2.5

C TODO
C  * Try using umlal or umaal.
C  * Try using ldm/stm.

define(`qp',	  `r0')
define(`up',	  `r1')
define(`n',	  `r2')
define(`bd',	  `r3')
define(`cy',	  `sp,#0')

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_bdiv_dbm1c)
	push	{r4, r5, r6, r7, r8}
	ldr	r4, [up], #4
	ldr	r5, [sp, #20]
	ands	r12, n, #3
	beq	L(fi0)
	cmp	r12, #2
	bcc	L(fi1)
	beq	L(fi2)

L(fi3):	umull	r8, r12, r4, bd
	ldr	r4, [up], #4
	b	L(lo3)

L(fi0):	umull	r6, r7, r4, bd
	ldr	r4, [up], #4
	b	L(lo0)

L(fi1):	subs	n, n, #1
	umull	r8, r12, r4, bd
	bls	L(wd1)
	ldr	r4, [up], #4
	b	L(lo1)

L(fi2):	umull	r6, r7, r4, bd
	ldr	r4, [up], #4
	b	L(lo2)

L(top):	ldr	r4, [up], #4
	subs	r5, r5, r6
	str	r5, [qp], #4
	sbc	r5, r5, r7
L(lo1):	umull	r6, r7, r4, bd
	ldr	r4, [up], #4
	subs	r5, r5, r8
	str	r5, [qp], #4
	sbc	r5, r5, r12
L(lo0):	umull	r8, r12, r4, bd
	ldr	r4, [up], #4
	subs	r5, r5, r6
	str	r5, [qp], #4
	sbc	r5, r5, r7
L(lo3):	umull	r6, r7, r4, bd
	ldr	r4, [up], #4
	subs	r5, r5, r8
	str	r5, [qp], #4
	sbc	r5, r5, r12
L(lo2):	subs	n, n, #4
	umull	r8, r12, r4, bd
	bhi	L(top)

L(wd2):	subs	r5, r5, r6
	str	r5, [qp], #4
	sbc	r5, r5, r7
L(wd1):	subs	r5, r5, r8
	str	r5, [qp]
	sbc	r0, r5, r12
	pop	{r4, r5, r6, r7, r8}
	bx	lr
EPILOGUE()
