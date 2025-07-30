dnl  SPARC v7 __udiv_qrnnd division support, used from longlong.h.
dnl  This is for v7 CPUs with a floating-point unit.

dnl  Copyright 1993, 1994, 1996, 2000 Free Software Foundation, Inc.

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
C rem_ptr	i0
C n1		i1
C n0		i2
C d		i3

ASM_START()

ifdef(`PIC',
`	TEXT
L(getpc):
	retl
	nop')

	TEXT
	ALIGN(8)
L(C0):	.double	0r4294967296
L(C1):	.double	0r2147483648

PROLOGUE(mpn_udiv_qrnnd)
	save	%sp,-104,%sp
	st	%i1,[%fp-8]
	ld	[%fp-8],%f10

ifdef(`PIC',
`L(pc):	call	L(getpc)		C put address of this insn in %o7
	ldd	[%o7+L(C0)-L(pc)],%f8',
`	sethi	%hi(L(C0)),%o7
	ldd	[%o7+%lo(L(C0))],%f8')

	fitod	%f10,%f4
	cmp	%i1,0
	bge	L(248)
	mov	%i0,%i5
	faddd	%f4,%f8,%f4
L(248):
	st	%i2,[%fp-8]
	ld	[%fp-8],%f10
	fmuld	%f4,%f8,%f6
	cmp	%i2,0
	bge	L(249)
	fitod	%f10,%f2
	faddd	%f2,%f8,%f2
L(249):
	st	%i3,[%fp-8]
	faddd	%f6,%f2,%f2
	ld	[%fp-8],%f10
	cmp	%i3,0
	bge	L(250)
	fitod	%f10,%f4
	faddd	%f4,%f8,%f4
L(250):
	fdivd	%f2,%f4,%f2

ifdef(`PIC',
`	ldd	[%o7+L(C1)-L(pc)],%f4',
`	sethi	%hi(L(C1)),%o7
	ldd	[%o7+%lo(L(C1))],%f4')

	fcmped	%f2,%f4
	nop
	fbge,a	L(251)
	fsubd	%f2,%f4,%f2
	fdtoi	%f2,%f2
	st	%f2,[%fp-8]
	b	L(252)
	ld	[%fp-8],%i4
L(251):
	fdtoi	%f2,%f2
	st	%f2,[%fp-8]
	ld	[%fp-8],%i4
	sethi	%hi(-2147483648),%g2
	xor	%i4,%g2,%i4
L(252):
	wr	%g0,%i4,%y
	sra	%i3,31,%g2
	and	%i4,%g2,%g2
	andcc	%g0,0,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,%i3,%g1
	mulscc	%g1,0,%g1
	add	%g1,%g2,%i0
	rd	%y,%g3
	subcc	%i2,%g3,%o7
	subxcc	%i1,%i0,%g0
	be	L(253)
	cmp	%o7,%i3

	add	%i4,-1,%i0
	add	%o7,%i3,%o7
	st	%o7,[%i5]
	ret
	restore
L(253):
	blu	L(246)
	mov	%i4,%i0
	add	%i4,1,%i0
	sub	%o7,%i3,%o7
L(246):
	st	%o7,[%i5]
	ret
	restore
EPILOGUE(mpn_udiv_qrnnd)
