dnl  ARM64 mpn_cnd_add_n, mpn_cnd_sub_n

dnl  Contributed to the GNU project by Torbjörn Granlund.

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
C Cortex-A53	 ?
C Cortex-A57	 ?

changecom(@&*$)

define(`cnd',	`x0')
define(`rp',	`x1')
define(`up',	`x2')
define(`vp',	`x3')
define(`n',	`x4')

ifdef(`OPERATION_cnd_add_n', `
  define(`ADDSUBC',      adcs)
  define(`CLRCY',	`cmn	xzr, xzr')
  define(`RETVAL',	`adc	x0, xzr, xzr')
  define(func,		mpn_cnd_add_n)')
ifdef(`OPERATION_cnd_sub_n', `
  define(`ADDSUBC',      sbcs)
  define(`CLRCY',	`cmp	xzr, xzr')
  define(`RETVAL',	`sbc	x0, xzr, xzr
			and	x0, x0, #1')
  define(func,		mpn_cnd_sub_n)')

MULFUNC_PROLOGUE(mpn_cnd_add_n mpn_cnd_sub_n)

ASM_START()
PROLOGUE(func)
	cmp	cnd, #1
	sbc	cnd, cnd, cnd

	CLRCY				C really only needed for n = 0 (mod 4)

	tbz	n, #0, L(1)
	ldr	x10, [up], #8
	ldr	x12, [vp], #8
	bic	x6, x12, cnd
	ADDSUBC	x8, x10, x6
	sub	n, n, #1
	str	x8, [rp], #8
	cbz	n, L(rt)

L(1):	ldp	x10, x11, [up], #16
	ldp	x12, x13, [vp], #16
	sub	n, n, #2
	cbz	n, L(end)

L(top):	bic	x6, x12, cnd
	bic	x7, x13, cnd
	ldp	x12, x13, [vp], #16
	ADDSUBC	x8, x10, x6
	ADDSUBC	x9, x11, x7
	ldp	x10, x11, [up], #16
	sub	n, n, #2
	stp	x8, x9, [rp], #16
	cbnz	n, L(top)

L(end):	bic	x6, x12, cnd
	bic	x7, x13, cnd
	ADDSUBC	x8, x10, x6
	ADDSUBC	x9, x11, x7
	stp	x8, x9, [rp]
L(rt):	RETVAL
	ret
EPILOGUE()
