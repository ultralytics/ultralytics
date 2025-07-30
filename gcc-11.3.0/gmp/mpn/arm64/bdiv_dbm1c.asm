dnl  ARM64 mpn_bdiv_dbm1c.

dnl  Copyright 2008, 2011, 2012, 2014 Free Software Foundation, Inc.

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

define(`qp',	  `x0')
define(`up',	  `x1')
define(`n',	  `x2')
define(`bd',	  `x3')
define(`cy',	  `x4')

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_bdiv_dbm1c)
	ldr	x5, [up], #8
	ands	x6, n, #3
	b.eq	L(fi0)
	cmp	x6, #2
	b.cc	L(fi1)
	b.eq	L(fi2)

L(fi3):	mul	x12, x5, bd
	umulh	x13, x5, bd
	ldr	x5, [up], #8
	b	L(lo3)

L(fi0):	mul	x10, x5, bd
	umulh	x11, x5, bd
	ldr	x5, [up], #8
	b	L(lo0)

L(fi1):	subs	n, n, #1
	mul	x12, x5, bd
	umulh	x13, x5, bd
	b.ls	L(wd1)
	ldr	x5, [up], #8
	b	L(lo1)

L(fi2):	mul	x10, x5, bd
	umulh	x11, x5, bd
	ldr	x5, [up], #8
	b	L(lo2)

L(top):	ldr	x5, [up], #8
	subs	x4, x4, x10
	str	x4, [qp], #8
	sbc	x4, x4, x11
L(lo1):	mul	x10, x5, bd
	umulh	x11, x5, bd
	ldr	x5, [up], #8
	subs	x4, x4, x12
	str	x4, [qp], #8
	sbc	x4, x4, x13
L(lo0):	mul	x12, x5, bd
	umulh	x13, x5, bd
	ldr	x5, [up], #8
	subs	x4, x4, x10
	str	x4, [qp], #8
	sbc	x4, x4, x11
L(lo3):	mul	x10, x5, bd
	umulh	x11, x5, bd
	ldr	x5, [up], #8
	subs	x4, x4, x12
	str	x4, [qp], #8
	sbc	x4, x4, x13
L(lo2):	subs	n, n, #4
	mul	x12, x5, bd
	umulh	x13, x5, bd
	b.hi	L(top)

L(wd2):	subs	x4, x4, x10
	str	x4, [qp], #8
	sbc	x4, x4, x11
L(wd1):	subs	x4, x4, x12
	str	x4, [qp]
	sbc	x0, x4, x13
	ret
EPILOGUE()
