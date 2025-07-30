dnl  ARM64 mpn_and_n, mpn_andn_n. mpn_nand_n, etc.

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
define(`vp', `x2')
define(`n',  `x3')

define(`POSTOP', `dnl')

ifdef(`OPERATION_and_n',`
  define(`func',    `mpn_and_n')
  define(`LOGOP',   `and	$1, $2, $3')')
ifdef(`OPERATION_andn_n',`
  define(`func',    `mpn_andn_n')
  define(`LOGOP',   `bic	$1, $2, $3')')
ifdef(`OPERATION_nand_n',`
  define(`func',    `mpn_nand_n')
  define(`POSTOP',  `mvn	$1, $1')
  define(`LOGOP',   `and	$1, $2, $3')')
ifdef(`OPERATION_ior_n',`
  define(`func',    `mpn_ior_n')
  define(`LOGOP',   `orr	$1, $2, $3')')
ifdef(`OPERATION_iorn_n',`
  define(`func',    `mpn_iorn_n')
  define(`LOGOP',   `orn	$1, $2, $3')')
ifdef(`OPERATION_nior_n',`
  define(`func',    `mpn_nior_n')
  define(`POSTOP',  `mvn	$1, $1')
  define(`LOGOP',   `orr	$1, $2, $3')')
ifdef(`OPERATION_xor_n',`
  define(`func',    `mpn_xor_n')
  define(`LOGOP',   `eor	$1, $2, $3')')
ifdef(`OPERATION_xnor_n',`
  define(`func',    `mpn_xnor_n')
  define(`LOGOP',   `eon	$1, $2, $3')')

MULFUNC_PROLOGUE(mpn_and_n mpn_andn_n mpn_nand_n mpn_ior_n mpn_iorn_n mpn_nior_n mpn_xor_n mpn_xnor_n)

ASM_START()
PROLOGUE(func)
	tbz	n, #0, L(b0)

	ldr	x4, [up],#8
	ldr	x6, [vp],#8
	sub	n, n, #1
	LOGOP(	x8, x4, x6)
	POSTOP(	x8)
	str	x8, [rp],#8
	cbz	n, L(rtn)

L(b0):	ldp	x4, x5, [up],#16
	ldp	x6, x7, [vp],#16
	sub	n, n, #2
	b	L(mid)

L(top):	ldp	x4, x5, [up],#16
	ldp	x6, x7, [vp],#16
	sub	n, n, #2
	stp	x8, x9, [rp],#16
L(mid):	LOGOP(	x8, x4, x6)
	LOGOP(	x9, x5, x7)
	POSTOP(	x8)
	POSTOP(	x9)
	cbnz	n, L(top)

	stp	x8, x9, [rp],#16
L(rtn):	ret
EPILOGUE()
