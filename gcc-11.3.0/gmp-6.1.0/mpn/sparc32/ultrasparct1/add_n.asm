dnl  SPARC T1 32-bit mpn_add_n.

dnl  Copyright 2010 Free Software Foundation, Inc.

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

C INPUT PARAMETERS
define(`rp',  %o0)
define(`ap',  %o1)
define(`bp',  %o2)
define(`n',   %o3)
define(`cy',  %o4)

define(`i',   %o3)

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc)

ASM_START()
PROLOGUE(mpn_add_nc)
	b	L(ent)
	srl	cy, 0, cy	C strip any bogus high bits
EPILOGUE()

PROLOGUE(mpn_add_n)
	mov	0, cy
L(ent):	srl	n, 0, n		C strip any bogus high bits
	sll	n, 2, n
	add	ap, n, ap
	add	bp, n, bp
	add	rp, n, rp
	neg	n, i

L(top):	lduw	[ap+i], %g1
	lduw	[bp+i], %g2
	add	%g1, %g2, %g3
	add	%g3, cy, %g3
	stw	%g3, [rp+i]
	add	i, 4, i
	brnz	i, L(top)
	srlx	%g3, 32, cy

	retl
	mov	cy, %o0		C return value
EPILOGUE()
