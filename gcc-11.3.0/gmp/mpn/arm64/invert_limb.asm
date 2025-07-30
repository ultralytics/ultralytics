dnl  ARM64 mpn_invert_limb -- Invert a normalized limb.

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

C            cycles/limb
C Cortex-A53     ?
C Cortex-A57     ?

C Compiler generated, mildly edited.  Could surely be further optimised.

ASM_START()
PROLOGUE(mpn_invert_limb)
	lsr	x2, x0, #54
	adrp	x1, approx_tab
	and	x2, x2, #0x1fe
	add	x1, x1, :lo12:approx_tab
	ldrh	w3, [x1,x2]
	lsr	x4, x0, #24
	add	x4, x4, #1
	ubfiz	x2, x3, #11, #16
	umull	x3, w3, w3
	mul	x3, x3, x4
	sub	x2, x2, #1
	sub	x2, x2, x3, lsr #40
	lsl	x3, x2, #60
	mul	x1, x2, x2
	msub	x1, x1, x4, x3
	lsl	x2, x2, #13
	add	x1, x2, x1, lsr #47
	and	x2, x0, #1
	neg	x3, x2
	and	x3, x3, x1, lsr #1
	add	x2, x2, x0, lsr #1
	msub	x2, x1, x2, x3
	umulh	x2, x2, x1
	lsl	x1, x1, #31
	add	x1, x1, x2, lsr #1
	mul	x3, x1, x0
	umulh	x2, x1, x0
	adds	x4, x3, x0
	adc	x0, x2, x0
	sub	x0, x1, x0
	ret
EPILOGUE()

	RODATA
	ALIGN(2)
	TYPE(   approx_tab, object)
	SIZE(   approx_tab, 512)
approx_tab:
forloop(i,256,512-1,dnl
`	.hword	eval(0x7fd00/i)
')dnl
