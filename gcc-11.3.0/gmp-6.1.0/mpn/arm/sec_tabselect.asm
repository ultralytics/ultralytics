dnl  ARM mpn_sec_tabselect

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

C	     cycles/limb
C StrongARM	 ?
C XScale	 ?
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	 2.33
C Cortex-A15	 2.2

C TODO
C  * Consider using special code for small nents, either swapping the inner and
C    outer loops, or providing a few completely unrolling the inner loops.

define(`rp',    `r0')
define(`tp',    `r1')
define(`n',     `r2')
define(`nents', `r3')
C      which  on stack

define(`i',     `r11')
define(`j',     `r12')
define(`c',     `r14')
define(`mask',  `r7')

ASM_START()
PROLOGUE(mpn_sec_tabselect)
	push	{r4-r11, r14}

	subs	j, n, #3
	bmi	L(outer_end)
L(outer_top):
	ldr	c, [sp, #36]
	mov	i, nents
	push	{tp}

	mov	r8, #0
	mov	r9, #0
	mov	r10, #0

L(top):	subs	c, c, #1
	ldm	tp, {r4,r5,r6}
	sbc	mask, mask, mask
	subs	i, i, #1
	add	tp, tp, n, lsl #2
	and	r4, r4, mask
	and	r5, r5, mask
	and	r6, r6, mask
	orr	r8, r8, r4
	orr	r9, r9, r5
	orr	r10, r10, r6
	bge	L(top)

	stmia	rp!, {r8,r9,r10}
	pop	{tp}
	add	tp, tp, #12
	subs	j, j, #3
	bpl	L(outer_top)
L(outer_end):

	cmp	j, #-1
	bne	L(n2)

	ldr	c, [sp, #36]
	mov	i, nents
	mov	r8, #0
	mov	r9, #0
L(tp2):	subs	c, c, #1
	sbc	mask, mask, mask
	ldm	tp, {r4,r5}
	subs	i, i, #1
	add	tp, tp, n, lsl #2
	and	r4, r4, mask
	and	r5, r5, mask
	orr	r8, r8, r4
	orr	r9, r9, r5
	bge	L(tp2)
	stmia	rp, {r8,r9}
	pop	{r4-r11, r14}
	bx	lr

L(n2):	cmp	j, #-2
	bne	L(n1)

	ldr	c, [sp, #36]
	mov	i, nents
	mov	r8, #0
L(tp1):	subs	c, c, #1
	sbc	mask, mask, mask
	ldr	r4, [tp]
	subs	i, i, #1
	add	tp, tp, n, lsl #2
	and	r4, r4, mask
	orr	r8, r8, r4
	bge	L(tp1)
	str	r8, [rp]
L(n1):	pop	{r4-r11, r14}
	bx	lr
EPILOGUE()
