dnl  ARM64 mpn_submul_1

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
C Cortex-A53	 ?
C Cortex-A57	 ?

changecom(@&*$)

define(`rp', `x0')
define(`up', `x1')
define(`n',  `x2')
define(`v0', `x3')

ifdef(`OPERATION_addmul_1', `
  define(`ADDSUB',	adds)
  define(`ADDSUBC',	adcs)
  define(`COND',	`cc')
  define(`func',	mpn_addmul_1)')
ifdef(`OPERATION_submul_1', `
  define(`ADDSUB',	subs)
  define(`ADDSUBC',	sbcs)
  define(`COND',	`cs')
  define(`func',	mpn_submul_1)')

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_submul_1)

PROLOGUE(func)
	mov	x15, #0

	tbz	n, #0, L(1)

	ldr	x4, [up],#8
	mul	x8, x4, v0
	umulh	x12, x4, v0
	adds	x8, x8, x15
	ldr	x4, [rp,#0]
	ADDSUB	x8, x4, x8
	csinc	x15, x12, x12, COND
	str	x8, [rp],#8

L(1):	tbz	n, #1, L(2)

	ldp	x4, x5, [up],#16
	mul	x8, x4, v0
	umulh	x12, x4, v0
	mul	x9, x5, v0
	umulh	x13, x5, v0
	adds	x8, x8, x15
	adcs	x9, x9, x12
	ldp	x4, x5, [rp,#0]
	adc	x15, x13, xzr
	sub	n, n, #1
	ADDSUB	x8, x4, x8
	ADDSUBC	x9, x5, x9
	csinc	x15, x15, x15, COND
	stp	x8, x9, [rp],#16

L(2):	lsr	n, n, #2
	cbz	n, L(end)

L(top):	ldp	x4, x5, [up],#16
	ldp	x6, x7, [up],#16
	mul	x8, x4, v0
	umulh	x12, x4, v0
	mul	x9, x5, v0
	umulh	x13, x5, v0
	adds	x8, x8, x15
	mul	x10, x6, v0
	umulh	x14, x6, v0
	adcs	x9, x9, x12
	mul	x11, x7, v0
	umulh	x15, x7, v0
	adcs	x10, x10, x13
	ldp	x4, x5, [rp,#0]
	adcs	x11, x11, x14
	ldp	x6, x7, [rp,#16]
	adc	x15, x15, xzr
	sub	n, n, #1
	ADDSUB	x8, x4, x8
	ADDSUBC	x9, x5, x9
	ADDSUBC	x10, x6, x10
	ADDSUBC	x11, x7, x11
	stp	x8, x9, [rp],#16
	csinc	x15, x15, x15, COND
	stp	x10, x11, [rp],#16
	cbnz	n, L(top)

L(end):	mov	x0, x15
	ret
EPILOGUE()
