dnl  ARM Neon mpn_sec_tabselect.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2011-2013 Free Software Foundation, Inc.

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
C StrongARM	 -
C XScale	 -
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	 1.15
C Cortex-A15	 0.65

define(`rp',     `r0')
define(`tp',     `r1')
define(`n',      `r2')
define(`nents',  `r3')
C define(`which',  on stack)

define(`i',      `r4')
define(`j',      `r5')

define(`maskq',  `q10')
define(`maskd',  `d20')

ASM_START()
PROLOGUE(mpn_sec_tabselect)
	push	{r4-r5}

	add	  r4, sp, #8
	vld1.32	  {d30[], d31[]}, [r4]	C 4 `which' copies
	vmov.i32  q14, #1		C 4 copies of 1

	subs	j, n, #8
	bmi	L(outer_end)

L(outer_top):
	mov	  i, nents
	mov	  r12, tp		C preserve tp
	veor	  q13, q13, q13		C 4 counter copies
	veor	  q2, q2, q2
	veor	  q3, q3, q3
	ALIGN(16)
L(top):	vceq.i32  maskq, q13, q15	C compare idx copies to `which' copies
	vld1.32	  {q0,q1}, [tp]
	vadd.i32  q13, q13, q14
	vbit	  q2, q0, maskq
	vbit	  q3, q1, maskq
	add	  tp, tp, n, lsl #2
	subs	  i, i, #1
	bne	  L(top)
	vst1.32	  {q2,q3}, [rp]!
	add	  tp, r12, #32		C restore tp, point to next slice
	subs	  j, j, #8
	bpl	  L(outer_top)
L(outer_end):

	tst	  n, #4
	beq	  L(b0xx)
L(b1xx):mov	  i, nents
	mov	  r12, tp
	veor	  q13, q13, q13
	veor	  q2, q2, q2
	ALIGN(16)
L(tp4):	vceq.i32  maskq, q13, q15
	vld1.32	  {q0}, [tp]
	vadd.i32  q13, q13, q14
	vbit	  q2, q0, maskq
	add	  tp, tp, n, lsl #2
	subs	  i, i, #1
	bne	  L(tp4)
	vst1.32	  {q2}, [rp]!
	add	  tp, r12, #16

L(b0xx):tst	  n, #2
	beq	  L(b00x)
L(b01x):mov	  i, nents
	mov	  r12, tp
	veor	  d26, d26, d26
	veor	  d4, d4, d4
	ALIGN(16)
L(tp2):	vceq.i32  maskd, d26, d30
	vld1.32	  {d0}, [tp]
	vadd.i32  d26, d26, d28
	vbit	  d4, d0, maskd
	add	  tp, tp, n, lsl #2
	subs	  i, i, #1
	bne	  L(tp2)
	vst1.32	  {d4}, [rp]!
	add	  tp, r12, #8

L(b00x):tst	  n, #1
	beq	  L(b000)
L(b001):mov	  i, nents
	mov	  r12, tp
	veor	  d26, d26, d26
	veor	  d4, d4, d4
	ALIGN(16)
L(tp1):	vceq.i32  maskd, d26, d30
	vld1.32	  {d0[0]}, [tp]
	vadd.i32  d26, d26, d28
	vbit	  d4, d0, maskd
	add	  tp, tp, n, lsl #2
	subs	  i, i, #1
	bne	  L(tp1)
	vst1.32	  {d4[0]}, [rp]

L(b000):pop	{r4-r5}
	bx	r14
EPILOGUE()
