dnl  HP-PA  __udiv_qrnnd division support, used from longlong.h.
dnl  This version runs fast on PA 7000 and later.

dnl  Copyright 1993, 1994, 2000, 2001, 2003 Free Software Foundation, Inc.

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
C rem_ptr	gr26
C n1		gr25
C n0		gr24
C d		gr23

C This file has caused a lot of trouble, since it demands PIC reference to
C static data, which triggers bugs in gas (at least version 2.7 through
C 2.11.2).  When the bug is triggered, many bogus relocs are generated.  The
C current solution is to stuff data right into the code, and refer it using
C absolute offsets.  Fragile to be sure, but nothing else seems to work.

ASM_START()
ifdef(`PIC',`',
`	RODATA
	INT64(0000, 0x43f00000, 0x0)	C 2^64
')

PROLOGUE(mpn_udiv_qrnnd)
C	.callinfo	frame=64,no_calls

	ldo		64(%r30),%r30

	stws		%r25,-16(0,%r30)	C n_hi
	stws		%r24,-12(0,%r30)	C n_lo

ifdef(`PIC',
`	bl		.+20,%r31
	dep		%r0,31,2,%r31
	.word	0x0				C padding for alignment
	.word	0x43f00000, 0x0			C 2^64
	ldo		4(%r31),%r31',
`	ldil		`L'%L(0000),%r31
	ldo		R%L(0000)(%r31),%r31')

	fldds		-16(0,%r30),%fr5
	stws		%r23,-12(0,%r30)
	comib,<=	0,%r25,L(1)
	fcnvxf,dbl,dbl	%fr5,%fr5
	fldds		0(0,%r31),%fr4
	fadd,dbl	%fr4,%fr5,%fr5

LDEF(1)
	fcpy,sgl	%fr0,%fr6L
	fldws		-12(0,%r30),%fr6R
	fcnvxf,dbl,dbl	%fr6,%fr4

	fdiv,dbl	%fr5,%fr4,%fr5

	fcnvfx,dbl,dbl	%fr5,%fr4
	fstws		%fr4R,-16(%r30)
	xmpyu		%fr4R,%fr6R,%fr6
	ldws		-16(%r30),%r28
	fstds		%fr6,-16(0,%r30)
	ldws		-12(0,%r30),%r21
	ldws		-16(0,%r30),%r20
	sub		%r24,%r21,%r22
	subb		%r25,%r20,%r20
	comib,=		0,%r20,L(2)
	ldo		-64(%r30),%r30

	add		%r22,%r23,%r22
	ldo		-1(%r28),%r28

LDEF(2)
	bv		0(%r2)
	stws		%r22,0(0,%r26)

EPILOGUE(mpn_udiv_qrnnd)
