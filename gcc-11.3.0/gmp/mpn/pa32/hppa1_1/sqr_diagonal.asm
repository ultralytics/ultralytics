dnl  HP-PA 1.1 32-bit mpn_sqr_diagonal.

dnl  Copyright 2001, 2002 Free Software Foundation, Inc.

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

C This code runs at 6 cycles/limb on the PA7100 and 2.5 cycles/limb on PA8x00.
C 2-way unrolling wouldn't help the PA7100; it could however bring times down
C to 2.0 cycles/limb for the PA8x00.

C INPUT PARAMETERS
define(`rp',`%r26')
define(`up',`%r25')
define(`n',`%r24')

ASM_START()
PROLOGUE(mpn_sqr_diagonal)
	ldo		4(rp),rp
	fldws,ma	4(up),%fr4r
	addib,=		-1,n,L(exit)
	xmpyu		%fr4r,%fr4r,%fr5

LDEF(loop)
	fldws,ma	4(up),%fr4r
	fstws		%fr5r,-4(rp)
	fstws,ma	%fr5l,8(rp)
	addib,<>	-1,n,L(loop)
	xmpyu		%fr4r,%fr4r,%fr5

LDEF(exit)
	fstws		%fr5r,-4(rp)
	bv		0(%r2)
	fstws		%fr5l,0(rp)
EPILOGUE(mpn_sqr_diagonal)
