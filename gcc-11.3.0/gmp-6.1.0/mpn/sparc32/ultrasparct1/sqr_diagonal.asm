dnl  SPARC T1 32-bit mpn_sqr_diagonal.

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
define(`rp',	`%o0')
define(`up',	`%o1')
define(`n',	`%o2')

ASM_START()
PROLOGUE(mpn_sqr_diagonal)
	deccc	n			C n--
	nop

L(top):	lduw	[up+0], %g1
	add	up, 4, up		C up++
	mulx	%g1, %g1, %g3
	stw	%g3, [rp+0]
	srlx	%g3, 32, %g4
	stw	%g4, [rp+4]
	add	rp, 8, rp		C rp += 2
	bnz	%icc, L(top)
	deccc	n			C n--

	retl
	nop
EPILOGUE()
