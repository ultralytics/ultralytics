dnl  ARM64 mpn_mod_34lsub1 -- remainder modulo 2^48-1.

dnl  Copyright 2012-2014 Free Software Foundation, Inc.

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
C Cortex-A53	 ?
C Cortex-A57	 ?

define(`ap',	x0)
define(`n',	x1)

changecom(@&*$)

C mp_limb_t mpn_mod_34lsub1 (mp_srcptr up, mp_size_t n)

C TODO
C  * An alternative inner loop which could run at 0.722 c/l:
C	adds	x8, x8, x2
C	adcs	x9, x9, x3
C	ldp	x2, x3, [ap, #-32]
C	adcs	x10, x10, x4
C	adc	x12, x12, xzr
C	adds	x8, x8, x5
C	ldp	x4, x5, [ap, #-16]
C	sub	n, n, #6
C	adcs	x9, x9, x6
C	adcs	x10, x10, x7
C	ldp	x6, x7, [ap], #48
C	adc	x12, x12, xzr
C	tbz	n, #63, L(top)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_mod_34lsub1)
	subs	n, n, #3
	mov	x8, #0
	b.lt	L(le2)			C n <= 2

	ldp	x2, x3, [ap, #0]
	ldr	x4, [ap, #16]
	add	ap, ap, #24
	subs	n, n, #3
	b.lt	L(sum)			C n <= 5
	cmn	x0, #0			C clear carry

L(top):	ldp	x5, x6, [ap, #0]
	ldr	x7, [ap, #16]
	add	ap, ap, #24
	sub	n, n, #3
	adcs	x2, x2, x5
	adcs	x3, x3, x6
	adcs	x4, x4, x7
	tbz	n, #63, L(top)

	adc	x8, xzr, xzr		C x8 <= 1

L(sum):	cmn	n, #2
	mov	x5, #0
	b.lo	1f
	ldr	x5, [ap], #8
1:	mov	x6, #0
	b.ls	1f
	ldr	x6, [ap], #8
1:	adds	x2, x2, x5
	adcs	x3, x3, x6
	adcs	x4, x4, xzr
	adc	x8, x8, xzr		C x8 <= 2

L(sum2):
	and	x0, x2, #0xffffffffffff
	add	x0, x0, x2, lsr #48
	add	x0, x0, x8

	lsl	x8, x3, #16
	and	x1, x8, #0xffffffffffff
	add	x0, x0, x1
	add	x0, x0, x3, lsr #32

	lsl	x8, x4, #32
	and	x1, x8, #0xffffffffffff
	add	x0, x0, x1
	add	x0, x0, x4, lsr #16
	ret

L(le2):	cmn	n, #1
	b.ne	L(1)
	ldp	x2, x3, [ap]
	mov	x4, #0
	b	L(sum2)
L(1):	ldr	x2, [ap]
	and	x0, x2, #0xffffffffffff
	add	x0, x0, x2, lsr #48
	ret
EPILOGUE()
