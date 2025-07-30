dnl  HP-PA 32-bit mpn_sqr_diagonal optimized for the PA8x00.

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

C This code runs at 6 cycles/limb on the PA7100 and 2 cycles/limb on PA8x00.
C The 2-way unrolling is actually not helping the PA7100.

C INPUT PARAMETERS
define(`rp',`%r26')
define(`up',`%r25')
define(`n',`%r24')

ASM_START()
PROLOGUE(mpn_sqr_diagonal)

	fldws,ma	4(up),%fr4r
	addib,=		-1,n,L(end1)
	ldo		4(rp),rp

	fldws,ma	4(up),%fr6r
	addib,=		-1,n,L(end2)
	xmpyu		%fr4r,%fr4r,%fr5

	fldws,ma	4(up),%fr4r
	addib,=		-1,n,L(end3)
	xmpyu		%fr6r,%fr6r,%fr7


LDEF(loop)
	fldws,ma	4(up),%fr6r
	fstws		%fr5r,-4(rp)
	fstws,ma	%fr5l,8(rp)
	addib,=		-1,n,L(exite)
	xmpyu		%fr4r,%fr4r,%fr5
	fldws,ma	4(up),%fr4r
	fstws		%fr7r,-4(rp)
	fstws,ma	%fr7l,8(rp)
	addib,<>	-1,n,L(loop)
	xmpyu		%fr6r,%fr6r,%fr7

LDEF(exito)
	fstws		%fr5r,-4(rp)
	fstws		%fr5l,0(rp)
	xmpyu		%fr4r,%fr4r,%fr5
	fstws		%fr7r,4(rp)
	fstws		%fr7l,8(rp)
	fstws,mb	%fr5r,12(rp)
	bv		0(%r2)
	fstws		%fr5l,4(rp)

LDEF(exite)
	fstws		%fr7r,-4(rp)
	fstws		%fr7l,0(rp)
	xmpyu		%fr6r,%fr6r,%fr7
	fstws		%fr5r,4(rp)
	fstws		%fr5l,8(rp)
	fstws,mb	%fr7r,12(rp)
	bv		0(%r2)
	fstws		%fr7l,4(rp)

LDEF(end1)
	xmpyu		%fr4r,%fr4r,%fr5
	fstws		%fr5r,-4(rp)
	bv		0(%r2)
	fstws,ma	%fr5l,8(rp)

LDEF(end2)
	xmpyu		%fr6r,%fr6r,%fr7
	fstws		%fr5r,-4(rp)
	fstws		%fr5l,0(rp)
	fstws		%fr7r,4(rp)
	bv		0(%r2)
	fstws		%fr7l,8(rp)

LDEF(end3)
	fstws		%fr5r,-4(rp)
	fstws		%fr5l,0(rp)
	xmpyu		%fr4r,%fr4r,%fr5
	fstws		%fr7r,4(rp)
	fstws		%fr7l,8(rp)
	fstws,mb	%fr5r,12(rp)
	bv		0(%r2)
	fstws		%fr5l,4(rp)
EPILOGUE(mpn_sqr_diagonal)
