dnl  ARM64 Neon mpn_sec_tabselect.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2011-2014 Free Software Foundation, Inc.

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

C void
C mpn_sec_tabselect (mp_ptr rp, mp_srcptr *tab,
C		     mp_size_t n, mp_size_t nents, mp_size_t which)

changecom(@&*$)

define(`rp',     `x0')
define(`tp',     `x1')
define(`n',      `x2')
define(`nents',  `x3')
define(`which',  `x4')

define(`i',      `x5')
define(`j',      `x6')

define(`maskq',  `v4')

ASM_START()
PROLOGUE(mpn_sec_tabselect)
	dup	v7.2d, x4			C 2 `which' copies

	mov	x10, #1
	dup	v6.2d, x10			C 2 copies of 1

	subs	j, n, #4
	b.mi	L(outer_end)

L(outer_top):
	mov	i, nents
	mov	x12, tp				C preserve tp
	movi	v5.16b, #0			C zero 2 counter copies
	movi	v2.16b, #0
	movi	v3.16b, #0
	ALIGN(16)
L(tp4):	cmeq	maskq.2d, v5.2d, v7.2d		C compare idx copies to `which' copies
	ld1	{v0.2d,v1.2d}, [tp]
	add	v5.2d, v5.2d, v6.2d
	bit	v2.16b, v0.16b, maskq.16b
	bit	v3.16b, v1.16b, maskq.16b
	add	tp, tp, n, lsl #3
	sub	i, i, #1
	cbnz	i, L(tp4)
	st1	{v2.2d,v3.2d}, [rp], #32
	add	tp, x12, #32			C restore tp, point to next slice
	subs	j, j, #4
	b.pl	L(outer_top)
L(outer_end):

	tbz	n, #1, L(b0x)
	mov	i, nents
	mov	x12, tp
	movi	v5.16b, #0			C zero 2 counter copies
	movi	v2.16b, #0
	ALIGN(16)
L(tp2):	cmeq	maskq.2d, v5.2d, v7.2d
	ld1	{v0.2d}, [tp]
	add	v5.2d, v5.2d, v6.2d
	bit	v2.16b, v0.16b, maskq.16b
	add	tp, tp, n, lsl #3
	sub	i, i, #1
	cbnz	i, L(tp2)
	st1	{v2.2d}, [rp], #16
	add	tp, x12, #16

L(b0x):	tbz	n, #0, L(b00)
	mov	i, nents
	mov	x12, tp
	movi	v5.16b, #0			C zero 2 counter copies
	movi	v2.16b, #0
	ALIGN(16)
L(tp1):	cmeq	maskq.2d, v5.2d, v7.2d
	ld1	{v0.1d}, [tp]
	add	v5.2d, v5.2d, v6.2d		C FIXME size should be `1d'
	bit	v2.8b, v0.8b, maskq.8b
	add	tp, tp, n, lsl #3
	sub	i, i, #1
	cbnz	i, L(tp1)
	st1	{v2.1d}, [rp], #8
	add	tp, x12, #8

L(b00):	ret
EPILOGUE()
